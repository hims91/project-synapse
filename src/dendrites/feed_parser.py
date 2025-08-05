"""
Dendrites - RSS/Atom Feed Parser
Layer 0: Sensory Input

This module implements comprehensive RSS 2.0 and Atom feed parsing
with validation, error handling, and metadata extraction.
Uses the improved URL resolver for Google News URL decoding.
"""
import asyncio
import base64
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, field
from enum import Enum
import structlog

# Import the improved URL resolver
from ..shared.url_resolver import resolve_final_url

logger = structlog.get_logger(__name__)


# DEPRECATED: Complex byte-parsing logic replaced with superior HTTP redirect approach
# See src/shared/url_resolver.py for the new, reliable implementation

def decode_google_read_url(google_read_url: str) -> Optional[str]:
    """
    Decodes a complex Google News "read" URL to find the original source article URL.
    This type of URL contains a Base64-encoded, protobuf-like data structure.
    
    This "deep" decoder is more robust for URLs found via the web interface.
    
    Args:
        google_read_url: The full URL from the browser, e.g., starting with news.google.com/read/...
        
    Returns:
        The clean, direct URL to the source article, or None if it cannot be found.
    """
    if "/read/CBM" not in google_read_url:
        # This decoder is specifically for the 'read' path structure
        return None
    
    # 1. Extract the core encoded string from the URL path
    match = re.search(r'/read/(CBMi[a-zA-Z0-9_-]+)', google_read_url)
    if not match:
        return None
    
    encoded_string = match.group(1)
    
    try:
        # 2. URL-decode the string (sometimes contains %xx characters)
        from urllib.parse import unquote
        url_decoded_string = unquote(encoded_string)
        
        # 3. Base64-decode the string
        # This requires proper padding. We'll add it to be safe.
        padding = '=' * (4 - len(url_decoded_string) % 4)
        decoded_bytes = base64.b64decode(url_decoded_string + padding)
        
        # 4. Use multiple strategies to find URLs in the binary data
        
        # Strategy 1: Look for complete HTTP/HTTPS URLs
        potential_urls = re.findall(rb'https?://[^\x00-\x20\x7f-\xff]+', decoded_bytes)
        
        if potential_urls:
            # Find the longest URL
            longest_url = max((url.decode('utf-8') for url in potential_urls), key=len)
            if len(longest_url) >= 20:
                return longest_url
        
        # Strategy 2: Look for domain patterns and try to reconstruct URLs
        # Convert to text with error handling
        try:
            text = decoded_bytes.decode('utf-8', errors='ignore')
            
            # Look for domain-like patterns
            domain_patterns = re.findall(r'([a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|co\.uk|co\.in|in|today|news|times)[a-zA-Z0-9/._-]*)', text)
            
            if domain_patterns:
                # Try to find the longest domain pattern and construct a URL
                longest_domain = max(domain_patterns, key=len)
                if len(longest_domain) >= 10:
                    # Construct HTTPS URL
                    constructed_url = f"https://{longest_domain}"
                    return constructed_url
        
        except Exception:
            pass
        
        # Strategy 3: Look for URL-like patterns without protocol
        url_patterns = re.findall(rb'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[a-zA-Z0-9/._-]*', decoded_bytes)
        
        if url_patterns:
            try:
                longest_pattern = max((pattern.decode('utf-8', errors='ignore') for pattern in url_patterns), key=len)
                if len(longest_pattern) >= 10:
                    return f"https://{longest_pattern}"
            except Exception:
                pass
        
        return None
    
    except Exception as e:
        # If any part of the decoding fails, it's not a valid link.
        logger.debug("Failed to decode Google read URL", error=str(e), url=google_read_url)
        return None


class FeedType(str, Enum):
    """Supported feed types."""
    RSS_2_0 = "rss_2.0"
    RSS_1_0 = "rss_1.0"
    ATOM_1_0 = "atom_1.0"
    UNKNOWN = "unknown"


class FeedValidationError(Exception):
    """Raised when feed validation fails."""
    pass


class FeedParsingError(Exception):
    """Raised when feed parsing fails."""
    pass


def decode_google_news_url(google_news_url: str) -> Optional[str]:
    """
    DEPRECATED: This function is kept for backward compatibility only.
    
    The complex byte-parsing approach has been replaced with a superior
    HTTP redirect following method in src/shared/url_resolver.py
    
    New approach benefits:
    - Future-proof: Works regardless of Google's encoding changes
    - Simple & Reliable: Uses standard HTTP redirect behavior  
    - Universal: Works for both /rss/articles/ and /read/ URLs
    
    Args:
        google_news_url: The URL found in the <link> tag of a Google News RSS item.
        
    Returns:
        The clean, direct URL to the source article, or None if it cannot be found.
    """
    logger.warning(
        "Using deprecated decode_google_news_url function. "
        "Please migrate to src/shared/url_resolver.resolve_final_url() "
        "for better reliability and future-proofing."
    )
    
    # Simple fallback implementation for RSS format only
    if "/rss/articles/" not in google_news_url:
        return None
    
    try:
        parsed_url = urlparse(google_news_url)
        path_segments = parsed_url.path.split('/')
        
        if len(path_segments) >= 4 and path_segments[1] == 'rss' and path_segments[2] == 'articles':
            encoded_string = path_segments[3]
            
            try:
                decoded_bytes = base64.b64decode(encoded_string + '==')
                url_match = re.search(rb'https?://[a-zA-Z0-9\-._~:/?#[\]@!&\'()*+,;=%]+', decoded_bytes)
                
                if url_match:
                    return url_match.group(0).decode('utf-8', errors='ignore')
            except Exception:
                pass
    
    except Exception:
        pass
    
    return None


@dataclass
class FeedItem:
    """Represents a single feed item/entry."""
    title: str
    link: str
    description: str = ""
    content: str = ""
    author: str = ""
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    guid: str = ""
    categories: List[str] = field(default_factory=list)
    enclosures: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Normalize URLs (URL decoding will be done async later)
        self.link = self._normalize_url(self.link)
        
        # Generate GUID if not provided
        if not self.guid:
            self.guid = self.link or f"{self.title}_{self.published_at}"
        
        # Clean up text content
        self.title = self._clean_text(self.title)
        self.description = self._clean_text(self.description)
        self.content = self._clean_content(self.content)  # Preserve HTML in content
        self.author = self._clean_text(self.author)
    
    async def resolve_url(self) -> None:
        """
        Resolve Google News URLs and other redirectors to their final destinations.
        This method should be called after creating FeedItem instances.
        """
        if self.link:
            resolved_url = await resolve_final_url(self.link)
            if resolved_url != self.link:
                logger.info("Resolved redirector URL", 
                           original=self.link, 
                           resolved=resolved_url)
                self.link = resolved_url
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL format."""
        if not url:
            return ""
        
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            return f"https://{url}"
        return url
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags (basic cleanup)
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def _clean_content(self, content: str) -> str:
        """Clean content while preserving HTML tags."""
        if not content:
            return ""
        
        # Only normalize whitespace, preserve HTML
        content = content.strip()
        
        return content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'link': self.link,
            'description': self.description,
            'content': self.content,
            'author': self.author,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'guid': self.guid,
            'categories': self.categories,
            'enclosures': self.enclosures,
            'metadata': self.metadata
        }


@dataclass
class FeedMetadata:
    """Represents feed metadata and channel information."""
    title: str
    link: str
    description: str = ""
    language: str = ""
    copyright: str = ""
    managing_editor: str = ""
    web_master: str = ""
    pub_date: Optional[datetime] = None
    last_build_date: Optional[datetime] = None
    generator: str = ""
    docs: str = ""
    cloud: Dict[str, str] = field(default_factory=dict)
    ttl: Optional[int] = None
    image: Dict[str, str] = field(default_factory=dict)
    text_input: Dict[str, str] = field(default_factory=dict)
    skip_hours: List[int] = field(default_factory=list)
    skip_days: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    feed_type: FeedType = FeedType.UNKNOWN
    version: str = ""
    encoding: str = "utf-8"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'link': self.link,
            'description': self.description,
            'language': self.language,
            'copyright': self.copyright,
            'managing_editor': self.managing_editor,
            'web_master': self.web_master,
            'pub_date': self.pub_date.isoformat() if self.pub_date else None,
            'last_build_date': self.last_build_date.isoformat() if self.last_build_date else None,
            'generator': self.generator,
            'docs': self.docs,
            'cloud': self.cloud,
            'ttl': self.ttl,
            'image': self.image,
            'text_input': self.text_input,
            'skip_hours': self.skip_hours,
            'skip_days': self.skip_days,
            'categories': self.categories,
            'feed_type': self.feed_type,
            'version': self.version,
            'encoding': self.encoding,
            'metadata': self.metadata
        }


@dataclass
class ParsedFeed:
    """Represents a completely parsed feed with metadata and items."""
    metadata: FeedMetadata
    items: List[FeedItem]
    raw_xml: str = ""
    parse_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.parse_time:
            self.parse_time = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metadata': self.metadata.to_dict(),
            'items': [item.to_dict() for item in self.items],
            'parse_time': self.parse_time.isoformat() if self.parse_time else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'item_count': len(self.items)
        }


class FeedParser:
    """
    Comprehensive RSS/Atom feed parser with validation and error handling.
    
    Features:
    - Support for RSS 2.0, RSS 1.0, and Atom 1.0 formats
    - Feed validation and error handling for malformed feeds
    - Metadata extraction and URL normalization
    - Robust date parsing with multiple format support
    - Content cleaning and sanitization
    - Google News URL decoding for effective news feed processing
    """
    
    # Common date formats found in feeds
    DATE_FORMATS = [
        "%a, %d %b %Y %H:%M:%S %z",      # RFC 2822
        "%a, %d %b %Y %H:%M:%S %Z",      # RFC 2822 with timezone name
        "%a, %d %b %Y %H:%M:%S",         # RFC 2822 without timezone
        "%Y-%m-%dT%H:%M:%S%z",           # ISO 8601 with timezone
        "%Y-%m-%dT%H:%M:%SZ",            # ISO 8601 UTC
        "%Y-%m-%dT%H:%M:%S",             # ISO 8601 without timezone
        "%Y-%m-%d %H:%M:%S",             # Simple datetime
        "%Y-%m-%d",                      # Date only
        "%d %b %Y %H:%M:%S %z",          # Alternative RFC format
        "%d %b %Y",                      # Date only alternative
    ]
    
    def __init__(self):
        self.logger = logger.bind(component="feed_parser")
    
    async def _resolve_item_urls(self, items: List[FeedItem]) -> None:
        """
        Resolve URLs for all feed items concurrently.
        
        Args:
            items: List of FeedItem instances to resolve URLs for
        """
        if not items:
            return
        
        self.logger.info("Resolving URLs for feed items", item_count=len(items))
        
        # Resolve URLs concurrently for better performance
        tasks = [item.resolve_url() for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("URL resolution completed", item_count=len(items))
    
    async def parse_feed(self, xml_content: str, feed_url: str = "") -> ParsedFeed:
        """
        Parse RSS/Atom feed from XML content.
        
        Args:
            xml_content: Raw XML content of the feed
            feed_url: URL of the feed (for URL resolution)
            
        Returns:
            ParsedFeed object with metadata and items
            
        Raises:
            FeedParsingError: If feed cannot be parsed
        """
        try:
            self.logger.info("Starting feed parsing", feed_url=feed_url)
            
            # Clean and prepare XML
            xml_content = self._clean_xml(xml_content)
            
            # Parse XML
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                raise FeedParsingError(f"Invalid XML: {str(e)}")
            
            # Detect feed type
            feed_type = self._detect_feed_type(root)
            self.logger.info("Feed type detected", feed_type=feed_type)
            
            # Parse based on feed type
            if feed_type == FeedType.RSS_2_0:
                return await self._parse_rss_2_0(root, xml_content, feed_url)
            elif feed_type == FeedType.RSS_1_0:
                return await self._parse_rss_1_0(root, xml_content, feed_url)
            elif feed_type == FeedType.ATOM_1_0:
                return await self._parse_atom_1_0(root, xml_content, feed_url)
            else:
                raise FeedParsingError(f"Unsupported feed type: {feed_type}")
                
        except Exception as e:
            self.logger.error("Feed parsing failed", error=str(e), feed_url=feed_url)
            if isinstance(e, (FeedParsingError, FeedValidationError)):
                raise
            raise FeedParsingError(f"Unexpected error during parsing: {str(e)}")
    
    def _clean_xml(self, xml_content: str) -> str:
        """Clean and prepare XML content for parsing."""
        # Remove BOM if present
        if xml_content.startswith('\ufeff'):
            xml_content = xml_content[1:]
        
        # Remove leading/trailing whitespace
        xml_content = xml_content.strip()
        
        # Fix common XML issues
        xml_content = xml_content.replace('&nbsp;', ' ')
        xml_content = xml_content.replace('&amp;amp;', '&amp;')
        
        return xml_content
    
    def _detect_feed_type(self, root: ET.Element) -> FeedType:
        """Detect the type of feed from the root element."""
        tag = root.tag.lower()
        
        if tag == 'rss':
            version = root.get('version', '')
            if version.startswith('2.'):
                return FeedType.RSS_2_0
            elif version.startswith('1.'):
                return FeedType.RSS_1_0
            else:
                return FeedType.RSS_2_0  # Default to RSS 2.0
        
        elif tag == 'feed' or tag.endswith('}feed'):
            # Check for Atom namespace (with or without namespace prefix)
            return FeedType.ATOM_1_0  # Assume Atom
        
        elif tag in ['rdf:rdf', 'rdf'] or tag.endswith('}rdf'):
            return FeedType.RSS_1_0
        
        return FeedType.UNKNOWN    

    async def _parse_rss_2_0(self, root: ET.Element, xml_content: str, feed_url: str) -> ParsedFeed:
        """Parse RSS 2.0 feed."""
        channel = root.find('channel')
        if channel is None:
            raise FeedParsingError("RSS 2.0 feed missing channel element")
        
        # Parse metadata
        metadata = self._parse_rss_metadata(channel, feed_url)
        metadata.feed_type = FeedType.RSS_2_0
        metadata.version = root.get('version', '2.0')
        
        # Parse items
        items = []
        for item_elem in channel.findall('item'):
            try:
                item = self._parse_rss_item(item_elem, feed_url)
                items.append(item)
            except Exception as e:
                self.logger.warning("Failed to parse RSS item", error=str(e))
        
        # Resolve URLs for all items
        await self._resolve_item_urls(items)
        
        return ParsedFeed(
            metadata=metadata,
            items=items,
            raw_xml=xml_content
        )
    
    async def _parse_rss_1_0(self, root: ET.Element, xml_content: str, feed_url: str) -> ParsedFeed:
        """Parse RSS 1.0 (RDF) feed."""
        # RSS 1.0 uses RDF structure
        channel = root.find('.//{http://purl.org/rss/1.0/}channel')
        if channel is None:
            # Try without namespace
            channel = root.find('.//channel')
        
        if channel is None:
            raise FeedParsingError("RSS 1.0 feed missing channel element")
        
        # Parse metadata
        metadata = self._parse_rss_1_0_metadata(channel, feed_url)
        metadata.feed_type = FeedType.RSS_1_0
        metadata.version = "1.0"
        
        # Parse items
        items = []
        for item_elem in root.findall('.//{http://purl.org/rss/1.0/}item'):
            try:
                item = self._parse_rss_item(item_elem, feed_url)
                items.append(item)
            except Exception as e:
                self.logger.warning("Failed to parse RSS 1.0 item", error=str(e))
        
        # Try without namespace if no items found
        if not items:
            for item_elem in root.findall('.//item'):
                try:
                    item = self._parse_rss_item(item_elem, feed_url)
                    items.append(item)
                except Exception as e:
                    self.logger.warning("Failed to parse RSS 1.0 item", error=str(e))
        
        # Resolve URLs for all items
        await self._resolve_item_urls(items)
        
        return ParsedFeed(
            metadata=metadata,
            items=items,
            raw_xml=xml_content
        )
    
    async def _parse_atom_1_0(self, root: ET.Element, xml_content: str, feed_url: str) -> ParsedFeed:
        """Parse Atom 1.0 feed."""
        # Parse metadata
        metadata = self._parse_atom_metadata(root, feed_url)
        metadata.feed_type = FeedType.ATOM_1_0
        metadata.version = "1.0"
        
        # Parse entries
        items = []
        for entry_elem in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
            try:
                item = self._parse_atom_entry(entry_elem, feed_url)
                items.append(item)
            except Exception as e:
                self.logger.warning("Failed to parse Atom entry", error=str(e))
        
        # Try without namespace if no entries found
        if not items:
            for entry_elem in root.findall('.//entry'):
                try:
                    item = self._parse_atom_entry(entry_elem, feed_url)
                    items.append(item)
                except Exception as e:
                    self.logger.warning("Failed to parse Atom entry", error=str(e))
        
        # Resolve URLs for all items
        await self._resolve_item_urls(items)
        
        return ParsedFeed(
            metadata=metadata,
            items=items,
            raw_xml=xml_content
        )
    
    def _parse_rss_metadata(self, channel: ET.Element, feed_url: str) -> FeedMetadata:
        """Parse RSS channel metadata."""
        def get_text(elem_name: str) -> str:
            elem = channel.find(elem_name)
            return elem.text.strip() if elem is not None and elem.text else ""
        
        def get_date(elem_name: str) -> Optional[datetime]:
            date_str = get_text(elem_name)
            return self._parse_date(date_str) if date_str else None
        
        # Parse image
        image = {}
        image_elem = channel.find('image')
        if image_elem is not None:
            image = {
                'url': get_text('url') if image_elem.find('url') is not None else "",
                'title': get_text('title') if image_elem.find('title') is not None else "",
                'link': get_text('link') if image_elem.find('link') is not None else "",
                'width': get_text('width') if image_elem.find('width') is not None else "",
                'height': get_text('height') if image_elem.find('height') is not None else "",
                'description': get_text('description') if image_elem.find('description') is not None else ""
            }
        
        # Parse categories
        categories = []
        for cat_elem in channel.findall('category'):
            if cat_elem.text:
                categories.append(cat_elem.text.strip())
        
        return FeedMetadata(
            title=get_text('title'),
            link=get_text('link'),
            description=get_text('description'),
            language=get_text('language'),
            copyright=get_text('copyright'),
            managing_editor=get_text('managingEditor'),
            web_master=get_text('webMaster'),
            pub_date=get_date('pubDate'),
            last_build_date=get_date('lastBuildDate'),
            generator=get_text('generator'),
            docs=get_text('docs'),
            ttl=int(get_text('ttl')) if get_text('ttl').isdigit() else None,
            image=image,
            categories=categories
        )
    
    def _parse_atom_metadata(self, feed: ET.Element, feed_url: str) -> FeedMetadata:
        """Parse Atom feed metadata."""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        def get_text(elem_name: str) -> str:
            elem = feed.find(f'atom:{elem_name}', ns)
            if elem is None:
                elem = feed.find(elem_name)  # Try without namespace
            return elem.text.strip() if elem is not None and elem.text else ""
        
        def get_date(elem_name: str) -> Optional[datetime]:
            date_str = get_text(elem_name)
            return self._parse_date(date_str) if date_str else None
        
        # Get link
        link = ""
        link_elem = feed.find('atom:link[@rel="alternate"]', ns)
        if link_elem is None:
            link_elem = feed.find('atom:link', ns)
        if link_elem is None:
            link_elem = feed.find('link')  # Try without namespace
        
        if link_elem is not None:
            link = link_elem.get('href', '')
        
        # Parse categories
        categories = []
        for cat_elem in feed.findall('atom:category', ns):
            term = cat_elem.get('term', '')
            if term:
                categories.append(term)
        
        # Try without namespace if no categories found
        if not categories:
            for cat_elem in feed.findall('category'):
                term = cat_elem.get('term', '')
                if term:
                    categories.append(term)
        
        return FeedMetadata(
            title=get_text('title'),
            link=link,
            description=get_text('subtitle') or get_text('summary'),
            language=get_text('language'),
            generator=get_text('generator'),
            pub_date=get_date('published'),
            last_build_date=get_date('updated'),
            categories=categories
        )
    
    def _parse_rss_1_0_metadata(self, channel: ET.Element, feed_url: str) -> FeedMetadata:
        """Parse RSS 1.0 channel metadata."""
        def get_text(elem_name: str) -> str:
            # Try with RSS 1.0 namespace first
            elem = channel.find(f'{{http://purl.org/rss/1.0/}}{elem_name}')
            if elem is None:
                # Try without namespace
                elem = channel.find(elem_name)
            return elem.text.strip() if elem is not None and elem.text else ""
        
        def get_date(elem_name: str) -> Optional[datetime]:
            date_str = get_text(elem_name)
            return self._parse_date(date_str) if date_str else None
        
        # Parse categories
        categories = []
        for cat_elem in channel.findall('category'):
            if cat_elem.text:
                categories.append(cat_elem.text.strip())
        
        return FeedMetadata(
            title=get_text('title'),
            link=get_text('link'),
            description=get_text('description'),
            language=get_text('language'),
            copyright=get_text('copyright'),
            managing_editor=get_text('managingEditor'),
            web_master=get_text('webMaster'),
            pub_date=get_date('pubDate'),
            last_build_date=get_date('lastBuildDate'),
            generator=get_text('generator'),
            docs=get_text('docs'),
            categories=categories
        )
    
    def _parse_rss_item(self, item: ET.Element, feed_url: str) -> FeedItem:
        """Parse RSS item."""
        def get_text(elem_name: str) -> str:
            # Try with RSS 1.0 namespace first
            elem = item.find(f'{{http://purl.org/rss/1.0/}}{elem_name}')
            if elem is None:
                # Try without namespace (RSS 2.0)
                elem = item.find(elem_name)
            return elem.text.strip() if elem is not None and elem.text else ""
        
        def get_date(elem_name: str) -> Optional[datetime]:
            date_str = get_text(elem_name)
            return self._parse_date(date_str) if date_str else None
        
        # Parse enclosures
        enclosures = []
        for enc_elem in item.findall('enclosure'):
            enclosure = {
                'url': enc_elem.get('url', ''),
                'type': enc_elem.get('type', ''),
                'length': enc_elem.get('length', '')
            }
            enclosures.append(enclosure)
        
        # Parse categories
        categories = []
        for cat_elem in item.findall('category'):
            if cat_elem.text:
                categories.append(cat_elem.text.strip())
        
        # Get content (try content:encoded first, then description)
        content = get_text('{http://purl.org/rss/1.0/modules/content/}encoded')
        if not content:
            content = get_text('content')
        
        return FeedItem(
            title=get_text('title'),
            link=get_text('link'),
            description=get_text('description'),
            content=content,
            author=get_text('author') or get_text('{http://purl.org/dc/elements/1.1/}creator'),
            published_at=get_date('pubDate') or get_date('{http://purl.org/dc/elements/1.1/}date'),
            guid=get_text('guid'),
            categories=categories,
            enclosures=enclosures
        )
    
    def _parse_atom_entry(self, entry: ET.Element, feed_url: str) -> FeedItem:
        """Parse Atom entry."""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        def get_text(elem_name: str) -> str:
            elem = entry.find(f'atom:{elem_name}', ns)
            if elem is None:
                elem = entry.find(elem_name)  # Try without namespace
            return elem.text.strip() if elem is not None and elem.text else ""
        
        def get_date(elem_name: str) -> Optional[datetime]:
            date_str = get_text(elem_name)
            return self._parse_date(date_str) if date_str else None
        
        # Get link
        link = ""
        link_elem = entry.find('atom:link[@rel="alternate"]', ns)
        if link_elem is None:
            link_elem = entry.find('atom:link', ns)
        if link_elem is None:
            link_elem = entry.find('link')  # Try without namespace
        
        if link_elem is not None:
            link = link_elem.get('href', '')
        
        # Get content
        content = ""
        content_elem = entry.find('atom:content', ns)
        if content_elem is None:
            content_elem = entry.find('content')
        
        if content_elem is not None:
            content = content_elem.text or ""
        
        # Get author
        author = ""
        author_elem = entry.find('atom:author/atom:name', ns)
        if author_elem is None:
            author_elem = entry.find('author/name')
        if author_elem is not None:
            author = author_elem.text or ""
        
        # Parse categories
        categories = []
        for cat_elem in entry.findall('atom:category', ns):
            term = cat_elem.get('term', '')
            if term:
                categories.append(term)
        
        # Try without namespace if no categories found
        if not categories:
            for cat_elem in entry.findall('category'):
                term = cat_elem.get('term', '')
                if term:
                    categories.append(term)
        
        return FeedItem(
            title=get_text('title'),
            link=link,
            description=get_text('summary'),
            content=content,
            author=author,
            published_at=get_date('published'),
            updated_at=get_date('updated'),
            guid=get_text('id'),
            categories=categories
        )
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string using multiple formats."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Try each format
        for fmt in self.DATE_FORMATS:
            try:
                dt = datetime.strptime(date_str, fmt)
                # If no timezone info, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Try parsing with dateutil if available
        try:
            from dateutil import parser
            dt = parser.parse(date_str)
            # If no timezone info, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ImportError, ValueError):
            pass
        
        self.logger.warning("Failed to parse date", date_str=date_str)
        return None
    
    def _resolve_url(self, url: str, base_url: str) -> str:
        """Resolve relative URL against base URL."""
        if not url:
            return ""
        
        # If URL is already absolute, return as-is
        if url.startswith(('http://', 'https://')):
            return url
        
        # Use urljoin to resolve relative URLs
        return urljoin(base_url, url)


class FeedValidator:
    """Validates feed content and structure."""
    
    def __init__(self):
        self.logger = logger.bind(component="feed_validator")
    
    def validate_feed(self, parsed_feed: ParsedFeed) -> List[str]:
        """
        Validate parsed feed and return list of validation errors.
        
        Args:
            parsed_feed: ParsedFeed object to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate metadata
        metadata_errors = self._validate_metadata(parsed_feed.metadata)
        errors.extend(metadata_errors)
        
        # Validate items
        for i, item in enumerate(parsed_feed.items):
            item_errors = self._validate_item(item, i)
            errors.extend(item_errors)
        
        # Check for minimum requirements
        if not parsed_feed.items:
            errors.append("Feed contains no items")
        
        return errors
    
    def _validate_metadata(self, metadata: FeedMetadata) -> List[str]:
        """Validate feed metadata."""
        errors = []
        
        if not metadata.title:
            errors.append("Feed title is required")
        
        if not metadata.link:
            errors.append("Feed link is required")
        
        if not metadata.description:
            errors.append("Feed description is required")
        
        # Validate URL format
        if metadata.link and not self._is_valid_url(metadata.link):
            errors.append(f"Invalid feed link URL: {metadata.link}")
        
        return errors
    
    def _validate_item(self, item: FeedItem, index: int) -> List[str]:
        """Validate feed item."""
        errors = []
        prefix = f"Item {index + 1}: "
        
        if not item.title and not item.description:
            errors.append(f"{prefix}Either title or description is required")
        
        if item.link and not self._is_valid_url(item.link):
            errors.append(f"{prefix}Invalid item link URL: {item.link}")
        
        return errors
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid HTTP/HTTPS URL."""
        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except Exception:
            return False


# Example usage and testing
if __name__ == '__main__':
    import asyncio
    
    async def test_google_news_decoding():
        """Test Google News URL decoding functionality."""
        # Real example URL from Google News RSS feed
        gnews_url = "https://news.google.com/rss/articles/CBMiYmh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3dvcmxkL2FzaWEtcGFjaWZpYy9qYXBhbi10by1yZWxlYXNlLW1vcmUtd2F0ZXItZnJvbS1mdWt1c2hpbWEtcGxhbnQtMjAyMy0xMi0xNS_SAQA?oc=5"
        
        print(f"Original Google News URL:\n{gnews_url}\n")
        
        real_url = decode_google_news_url(gnews_url)
        
        if real_url:
            print(f"Decoded Source URL:\n{real_url}")
        else:
            print("Failed to decode the URL.")
    
    async def test_feed_parsing():
        """Test basic feed parsing functionality."""
        # Sample RSS 2.0 feed
        sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <link>https://example.com</link>
                <description>A test RSS feed</description>
                <item>
                    <title>Test Article</title>
                    <link>https://news.google.com/rss/articles/CBMiYmh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3dvcmxkL2FzaWEtcGFjaWZpYy9qYXBhbi10by1yZWxlYXNlLW1vcmUtd2F0ZXItZnJvbS1mdWt1c2hpbWEtcGxhbnQtMjAyMy0xMi0xNS_SAQA?oc=5</link>
                    <description>Test article description</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""
        
        parser = FeedParser()
        try:
            parsed_feed = await parser.parse_feed(sample_rss)
            print(f"Parsed feed: {parsed_feed.metadata.title}")
            print(f"Items: {len(parsed_feed.items)}")
            
            if parsed_feed.items:
                item = parsed_feed.items[0]
                print(f"First item title: {item.title}")
                print(f"First item link (decoded): {item.link}")
        
        except Exception as e:
            print(f"Parsing failed: {e}")
    
    # Run tests
    print("Testing Google News URL decoding...")
    asyncio.run(test_google_news_decoding())
    
    print("\n" + "="*50 + "\n")
    
    print("Testing feed parsing with Google News URL...")
    asyncio.run(test_feed_parsing())