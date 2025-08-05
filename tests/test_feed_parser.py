"""
Unit tests for RSS/Atom feed parser.
Tests feed parsing, validation, and error handling for various feed formats.
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

from src.dendrites.feed_parser import (
    FeedParser, FeedValidator, FeedType, FeedItem, FeedMetadata, ParsedFeed,
    FeedParsingError, FeedValidationError, decode_google_news_url
)


class TestFeedItem:
    """Test FeedItem functionality."""
    
    def test_feed_item_initialization(self):
        """Test FeedItem initialization and post-processing."""
        item = FeedItem(
            title="  Test Article  ",
            link="example.com/article",
            description="<p>Test description</p>",
            author="John Doe"
        )
        
        assert item.title == "Test Article"
        assert item.link == "https://example.com/article"
        assert item.description == "Test description"
        assert item.author == "John Doe"
        assert item.guid == "https://example.com/article"
    
    def test_feed_item_with_guid(self):
        """Test FeedItem with explicit GUID."""
        item = FeedItem(
            title="Test Article",
            link="https://example.com/article",
            guid="unique-id-123"
        )
        
        assert item.guid == "unique-id-123"
    
    def test_feed_item_to_dict(self):
        """Test FeedItem serialization."""
        published_at = datetime.now(timezone.utc)
        item = FeedItem(
            title="Test Article",
            link="https://example.com/article",
            description="Test description",
            published_at=published_at,
            categories=["tech", "news"]
        )
        
        item_dict = item.to_dict()
        
        assert item_dict['title'] == "Test Article"
        assert item_dict['link'] == "https://example.com/article"
        assert item_dict['published_at'] == published_at.isoformat()
        assert item_dict['categories'] == ["tech", "news"]


class TestFeedMetadata:
    """Test FeedMetadata functionality."""
    
    def test_feed_metadata_initialization(self):
        """Test FeedMetadata initialization."""
        metadata = FeedMetadata(
            title="Test Feed",
            link="https://example.com",
            description="Test feed description",
            language="en-US"
        )
        
        assert metadata.title == "Test Feed"
        assert metadata.link == "https://example.com"
        assert metadata.description == "Test feed description"
        assert metadata.language == "en-US"
        assert metadata.feed_type == FeedType.UNKNOWN
    
    def test_feed_metadata_to_dict(self):
        """Test FeedMetadata serialization."""
        pub_date = datetime.now(timezone.utc)
        metadata = FeedMetadata(
            title="Test Feed",
            link="https://example.com",
            description="Test description",
            pub_date=pub_date,
            feed_type=FeedType.RSS_2_0
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict['title'] == "Test Feed"
        assert metadata_dict['pub_date'] == pub_date.isoformat()
        assert metadata_dict['feed_type'] == FeedType.RSS_2_0


class TestFeedParser:
    """Test FeedParser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create FeedParser instance."""
        return FeedParser()
    
    @pytest.fixture
    def sample_rss_2_0(self):
        """Sample RSS 2.0 feed."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test RSS Feed</title>
                <link>https://example.com</link>
                <description>A test RSS feed</description>
                <language>en-US</language>
                <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                <lastBuildDate>Mon, 01 Jan 2024 12:00:00 GMT</lastBuildDate>
                <generator>Test Generator</generator>
                <ttl>60</ttl>
                
                <item>
                    <title>First Article</title>
                    <link>https://example.com/article1</link>
                    <description>Description of first article</description>
                    <author>john@example.com (John Doe)</author>
                    <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
                    <guid>https://example.com/article1</guid>
                    <category>Technology</category>
                    <category>News</category>
                </item>
                
                <item>
                    <title>Second Article</title>
                    <link>https://example.com/article2</link>
                    <description>Description of second article</description>
                    <pubDate>Mon, 01 Jan 2024 11:00:00 GMT</pubDate>
                    <guid>article-2-guid</guid>
                </item>
            </channel>
        </rss>'''
    
    @pytest.fixture
    def sample_atom_1_0(self):
        """Sample Atom 1.0 feed."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <title>Test Atom Feed</title>
            <link href="https://example.com"/>
            <updated>2024-01-01T12:00:00Z</updated>
            <author>
                <name>John Doe</name>
            </author>
            <id>https://example.com/feed</id>
            <subtitle>A test Atom feed</subtitle>
            
            <entry>
                <title>First Entry</title>
                <link href="https://example.com/entry1"/>
                <id>https://example.com/entry1</id>
                <updated>2024-01-01T10:00:00Z</updated>
                <published>2024-01-01T10:00:00Z</published>
                <summary>Summary of first entry</summary>
                <content type="html">Content of first entry</content>
                <author>
                    <name>Jane Smith</name>
                </author>
                <category term="technology"/>
            </entry>
            
            <entry>
                <title>Second Entry</title>
                <link href="https://example.com/entry2"/>
                <id>https://example.com/entry2</id>
                <updated>2024-01-01T11:00:00Z</updated>
                <summary>Summary of second entry</summary>
            </entry>
        </feed>'''
    
    @pytest.fixture
    def sample_rss_1_0(self):
        """Sample RSS 1.0 feed."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns="http://purl.org/rss/1.0/"
                 xmlns:dc="http://purl.org/dc/elements/1.1/">
            <channel rdf:about="https://example.com">
                <title>Test RSS 1.0 Feed</title>
                <link>https://example.com</link>
                <description>A test RSS 1.0 feed</description>
            </channel>
            
            <item rdf:about="https://example.com/item1">
                <title>First Item</title>
                <link>https://example.com/item1</link>
                <description>Description of first item</description>
                <dc:creator>John Doe</dc:creator>
                <dc:date>2024-01-01T10:00:00Z</dc:date>
            </item>
        </rdf:RDF>'''
    
    @pytest.mark.asyncio
    async def test_detect_feed_type_rss_2_0(self, parser):
        """Test RSS 2.0 feed type detection."""
        xml = '<rss version="2.0"><channel></channel></rss>'
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        
        feed_type = parser._detect_feed_type(root)
        assert feed_type == FeedType.RSS_2_0
    
    @pytest.mark.asyncio
    async def test_detect_feed_type_atom(self, parser):
        """Test Atom feed type detection."""
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        
        feed_type = parser._detect_feed_type(root)
        assert feed_type == FeedType.ATOM_1_0
    
    @pytest.mark.asyncio
    async def test_parse_rss_2_0_feed(self, parser, sample_rss_2_0):
        """Test parsing RSS 2.0 feed."""
        parsed_feed = await parser.parse_feed(sample_rss_2_0, "https://example.com/feed.xml")
        
        # Check metadata
        assert parsed_feed.metadata.title == "Test RSS Feed"
        assert parsed_feed.metadata.link == "https://example.com"
        assert parsed_feed.metadata.description == "A test RSS feed"
        assert parsed_feed.metadata.language == "en-US"
        assert parsed_feed.metadata.feed_type == FeedType.RSS_2_0
        assert parsed_feed.metadata.version == "2.0"
        assert parsed_feed.metadata.generator == "Test Generator"
        assert parsed_feed.metadata.ttl == 60
        
        # Check items
        assert len(parsed_feed.items) == 2
        
        first_item = parsed_feed.items[0]
        assert first_item.title == "First Article"
        assert first_item.link == "https://example.com/article1"
        assert first_item.description == "Description of first article"
        assert first_item.guid == "https://example.com/article1"
        assert first_item.categories == ["Technology", "News"]
        assert first_item.published_at is not None
        
        second_item = parsed_feed.items[1]
        assert second_item.title == "Second Article"
        assert second_item.guid == "article-2-guid"
    
    @pytest.mark.asyncio
    async def test_parse_atom_1_0_feed(self, parser, sample_atom_1_0):
        """Test parsing Atom 1.0 feed."""
        parsed_feed = await parser.parse_feed(sample_atom_1_0, "https://example.com/feed.xml")
        
        # Check metadata
        assert parsed_feed.metadata.title == "Test Atom Feed"
        assert parsed_feed.metadata.link == "https://example.com"
        assert parsed_feed.metadata.description == "A test Atom feed"
        assert parsed_feed.metadata.feed_type == FeedType.ATOM_1_0
        assert parsed_feed.metadata.version == "1.0"
        
        # Check items
        assert len(parsed_feed.items) == 2
        
        first_entry = parsed_feed.items[0]
        assert first_entry.title == "First Entry"
        assert first_entry.link == "https://example.com/entry1"
        assert first_entry.description == "Summary of first entry"
        assert first_entry.content == "Content of first entry"
        assert first_entry.author == "Jane Smith"
        assert first_entry.guid == "https://example.com/entry1"
        assert first_entry.categories == ["technology"]
        assert first_entry.published_at is not None
        assert first_entry.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_parse_rss_1_0_feed(self, parser, sample_rss_1_0):
        """Test parsing RSS 1.0 feed."""
        parsed_feed = await parser.parse_feed(sample_rss_1_0, "https://example.com/feed.xml")
        
        # Check metadata
        assert parsed_feed.metadata.title == "Test RSS 1.0 Feed"
        assert parsed_feed.metadata.link == "https://example.com"
        assert parsed_feed.metadata.description == "A test RSS 1.0 feed"
        assert parsed_feed.metadata.feed_type == FeedType.RSS_1_0
        assert parsed_feed.metadata.version == "1.0"
        
        # Check items
        assert len(parsed_feed.items) == 1
        
        item = parsed_feed.items[0]
        assert item.title == "First Item"
        assert item.link == "https://example.com/item1"
        assert item.description == "Description of first item"
        assert item.author == "John Doe"
        assert item.published_at is not None
    
    @pytest.mark.asyncio
    async def test_parse_invalid_xml(self, parser):
        """Test parsing invalid XML."""
        invalid_xml = "This is not XML"
        
        with pytest.raises(FeedParsingError, match="Invalid XML"):
            await parser.parse_feed(invalid_xml)
    
    @pytest.mark.asyncio
    async def test_parse_unsupported_feed_type(self, parser):
        """Test parsing unsupported feed type."""
        unsupported_xml = '<unknown><content>test</content></unknown>'
        
        with pytest.raises(FeedParsingError, match="Unsupported feed type"):
            await parser.parse_feed(unsupported_xml)
    
    @pytest.mark.asyncio
    async def test_parse_rss_without_channel(self, parser):
        """Test parsing RSS without channel element."""
        invalid_rss = '<rss version="2.0"><item><title>Test</title></item></rss>'
        
        with pytest.raises(FeedParsingError, match="RSS 2.0 feed missing channel element"):
            await parser.parse_feed(invalid_rss)
    
    @pytest.mark.asyncio
    async def test_parse_feed_with_malformed_items(self, parser):
        """Test parsing feed with some malformed items."""
        rss_with_bad_item = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <link>https://example.com</link>
                <description>Test description</description>
                
                <item>
                    <title>Good Item</title>
                    <link>https://example.com/good</link>
                    <description>Good description</description>
                </item>
                
                <item>
                    <!-- This item has issues but should not break parsing -->
                    <title></title>
                    <link>invalid-url</link>
                </item>
            </channel>
        </rss>'''
        
        parsed_feed = await parser.parse_feed(rss_with_bad_item)
        
        # Should parse successfully but with errors
        assert len(parsed_feed.items) >= 1  # At least the good item
        assert len(parsed_feed.errors) == 0  # Malformed items don't cause errors, just warnings
    
    def test_date_parsing(self, parser):
        """Test date parsing with various formats."""
        # RFC 2822 format
        date1 = parser._parse_date("Mon, 01 Jan 2024 12:00:00 GMT")
        assert date1 is not None
        assert date1.year == 2024
        
        # ISO 8601 format
        date2 = parser._parse_date("2024-01-01T12:00:00Z")
        assert date2 is not None
        assert date2.year == 2024
        
        # Invalid date
        date3 = parser._parse_date("invalid date")
        assert date3 is None
        
        # Empty date
        date4 = parser._parse_date("")
        assert date4 is None
    
    def test_url_resolution(self, parser):
        """Test URL resolution."""
        # Absolute URL
        url1 = parser._resolve_url("https://example.com/page", "https://base.com")
        assert url1 == "https://example.com/page"
        
        # Relative URL
        url2 = parser._resolve_url("/page", "https://base.com/feed")
        assert url2 == "https://base.com/page"
        
        # Empty URL
        url3 = parser._resolve_url("", "https://base.com")
        assert url3 == ""
        
        # No base URL
        url4 = parser._resolve_url("relative", "")
        assert url4 == "relative"
    
    def test_xml_cleaning(self, parser):
        """Test XML content cleaning."""
        # BOM removal
        xml_with_bom = '\ufeff<?xml version="1.0"?><root></root>'
        cleaned = parser._clean_xml(xml_with_bom)
        assert not cleaned.startswith('\ufeff')
        
        # Whitespace removal
        xml_with_whitespace = '  \n  <?xml version="1.0"?><root></root>  \n  '
        cleaned = parser._clean_xml(xml_with_whitespace)
        assert cleaned.startswith('<?xml')
        assert cleaned.endswith('</root>')
        
        # Entity replacement
        xml_with_entities = '<?xml version="1.0"?><root>&nbsp;&amp;amp;</root>'
        cleaned = parser._clean_xml(xml_with_entities)
        assert '&nbsp;' not in cleaned
        assert '&amp;amp;' not in cleaned


class TestFeedValidator:
    """Test FeedValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create FeedValidator instance."""
        return FeedValidator()
    
    def test_validate_valid_feed(self, validator):
        """Test validation of valid feed."""
        metadata = FeedMetadata(
            title="Valid Feed",
            link="https://example.com",
            description="Valid description"
        )
        
        items = [
            FeedItem(
                title="Valid Item",
                link="https://example.com/item1",
                description="Valid description"
            )
        ]
        
        parsed_feed = ParsedFeed(metadata=metadata, items=items)
        errors = validator.validate_feed(parsed_feed)
        
        assert len(errors) == 0
    
    def test_validate_feed_missing_title(self, validator):
        """Test validation of feed missing title."""
        metadata = FeedMetadata(
            title="",  # Missing title
            link="https://example.com",
            description="Valid description"
        )
        
        items = [FeedItem(title="Item", link="", description="Description")]
        parsed_feed = ParsedFeed(metadata=metadata, items=items)
        errors = validator.validate_feed(parsed_feed)
        
        assert any("title is required" in error for error in errors)
    
    def test_validate_feed_invalid_link(self, validator):
        """Test validation of feed with invalid link."""
        metadata = FeedMetadata(
            title="Valid Title",
            link="not-a-valid-url",  # Invalid URL
            description="Valid description"
        )
        
        items = [FeedItem(title="Item", link="invalid-url", description="Description")]
        parsed_feed = ParsedFeed(metadata=metadata, items=items)
        errors = validator.validate_feed(parsed_feed)
        
        assert any("Invalid" in error and "URL" in error for error in errors)
    
    def test_validate_empty_feed(self, validator):
        """Test validation of feed with no items."""
        metadata = FeedMetadata(
            title="Valid Title",
            link="https://example.com",
            description="Valid description"
        )
        
        parsed_feed = ParsedFeed(metadata=metadata, items=[])
        errors = validator.validate_feed(parsed_feed)
        
        assert any("contains no items" in error for error in errors)
    
    def test_validate_item_missing_content(self, validator):
        """Test validation of item missing both title and description."""
        metadata = FeedMetadata(
            title="Valid Title",
            link="https://example.com",
            description="Valid description"
        )
        
        items = [
            FeedItem(title="", description="", link="https://example.com/item1")
        ]
        
        parsed_feed = ParsedFeed(metadata=metadata, items=items)
        errors = validator.validate_feed(parsed_feed)
        
        assert any("Either title or description is required" in error for error in errors)
    
    def test_url_validation(self, validator):
        """Test URL validation."""
        # Valid URLs
        assert validator._is_valid_url("https://example.com")
        assert validator._is_valid_url("http://example.com/path")
        assert validator._is_valid_url("https://sub.example.com:8080/path?query=1")
        
        # Invalid URLs
        assert not validator._is_valid_url("not-a-url")
        assert not validator._is_valid_url("ftp://example.com")  # Wrong scheme
        assert not validator._is_valid_url("https://")  # Missing netloc
        assert not validator._is_valid_url("")  # Empty


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_parse_feed_method(self):
        """Test FeedParser.parse_feed method."""
        rss_xml = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <link>https://example.com</link>
                <description>Test description</description>
                <item>
                    <title>Test Item</title>
                    <link>https://example.com/item</link>
                    <description>Test item description</description>
                </item>
            </channel>
        </rss>'''
        
        parser = FeedParser()
        parsed_feed = await parser.parse_feed(rss_xml, "https://example.com/feed.xml")
        
        assert parsed_feed.metadata.title == "Test Feed"
        assert len(parsed_feed.items) == 1
        assert parsed_feed.items[0].title == "Test Item"
    
    def test_validate_parsed_feed(self):
        """Test FeedValidator.validate_feed method."""
        metadata = FeedMetadata(
            title="Test Feed",
            link="https://example.com",
            description="Test description"
        )
        
        items = [
            FeedItem(title="Test Item", link="https://example.com/item", description="Test description")
        ]
        
        parsed_feed = ParsedFeed(metadata=metadata, items=items)
        validator = FeedValidator()
        errors = validator.validate_feed(parsed_feed)
        
        assert len(errors) == 0


    def test_google_news_url_decoding(self):
        """Test Google News URL decoding functionality."""
        # Test with a real Google News RSS URL (fully supported)
        rss_url = "https://news.google.com/rss/articles/CBMiYmh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3dvcmxkL2FzaWEtcGFjaWZpYy9qYXBhbi10by1yZWxlYXNlLW1vcmUtd2F0ZXItZnJvbS1mdWt1c2hpbWEtcGxhbnQtMjAyMy0xMi0xNS_SAQA?oc=5"
        
        decoded_url = decode_google_news_url(rss_url)
        
        assert decoded_url is not None
        assert decoded_url.startswith("https://www.reuters.com")
        assert "fukushima" in decoded_url.lower()
        
        # Test with Google News /read/ URL (experimental support)
        read_url = "https://news.google.com/read/CBMiowJBVV95cUxPcHhpODBMVFJINkRHRk5pamRNY3BDVHZQS0MzekZucmJBNklYRlM3bXRaWmhHWjhod2tMUkhJSnUzVjc3Zll5OEszY0FETnU5V2s5SV9LRjVhaUpXUENUU1JlZmNHcTl2ZlEwTVhJa2FCdWVWSlhJSkxZZ0pFRU1fUEswV25wTV9uX21oUURXaFJXVldzaEg4bXZTcWpuWEVKXzYtNGN6anJWUE11TGUxQk9Cd1RmV0dnRmEzUXc5akl2dnJYQ0dNZnNucEZUcDVqQlpieTFlTlc5RVFGVUc0cmw3YjM4US1Pd0JlQXFQUWdySFlEdi1rVURDVnJWNl96bG8tY082X3NSaFBFRkJPS2pxNmVaTS0tOVlfSW01WE1McW8?hl=en-IN&gl=IN&ceid=IN%3Aen"
        
        decoded_read_url = decode_google_news_url(read_url)
        # Note: This may return None due to complex encoding - that's expected
        # The test documents the current limitation
        
        # Test with non-Google News URL
        regular_url = "https://example.com/article"
        decoded_regular = decode_google_news_url(regular_url)
        assert decoded_regular is None
        
        # Test with invalid URL
        invalid_url = "not-a-url"
        decoded_invalid = decode_google_news_url(invalid_url)
        assert decoded_invalid is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def parser(self):
        """Create FeedParser instance."""
        return FeedParser()
    
    @pytest.mark.asyncio
    async def test_feed_with_namespaces(self, parser):
        """Test feed with various namespaces."""
        namespaced_rss = '''<?xml version="1.0"?>
        <rss version="2.0" 
             xmlns:content="http://purl.org/rss/1.0/modules/content/"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
            <channel>
                <title>Namespaced Feed</title>
                <link>https://example.com</link>
                <description>Feed with namespaces</description>
                
                <item>
                    <title>Namespaced Item</title>
                    <link>https://example.com/item</link>
                    <description>Basic description</description>
                    <content:encoded><![CDATA[<p>Rich content</p>]]></content:encoded>
                    <dc:creator>John Doe</dc:creator>
                </item>
            </channel>
        </rss>'''
        
        parsed_feed = await parser.parse_feed(namespaced_rss)
        
        assert parsed_feed.metadata.title == "Namespaced Feed"
        assert len(parsed_feed.items) == 1
        
        item = parsed_feed.items[0]
        assert item.title == "Namespaced Item"
        assert item.content == "<p>Rich content</p>"  # Should extract content:encoded
        assert item.author == "John Doe"  # Should extract dc:creator
    
    @pytest.mark.asyncio
    async def test_feed_with_cdata(self, parser):
        """Test feed with CDATA sections."""
        cdata_rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title><![CDATA[Feed with CDATA]]></title>
                <link>https://example.com</link>
                <description><![CDATA[Description with <em>HTML</em>]]></description>
                
                <item>
                    <title><![CDATA[Item with CDATA]]></title>
                    <link>https://example.com/item</link>
                    <description><![CDATA[Description with <strong>formatting</strong>]]></description>
                </item>
            </channel>
        </rss>'''
        
        parsed_feed = await parser.parse_feed(cdata_rss)
        
        assert parsed_feed.metadata.title == "Feed with CDATA"
        assert "HTML" in parsed_feed.metadata.description
        
        item = parsed_feed.items[0]
        assert item.title == "Item with CDATA"
        assert "formatting" in item.description
    
    @pytest.mark.asyncio
    async def test_atom_without_namespace(self, parser):
        """Test Atom feed without proper namespace."""
        atom_no_ns = '''<?xml version="1.0"?>
        <feed>
            <title>Atom without namespace</title>
            <link href="https://example.com"/>
            <updated>2024-01-01T12:00:00Z</updated>
            <id>https://example.com/feed</id>
            
            <entry>
                <title>Entry without namespace</title>
                <link href="https://example.com/entry"/>
                <id>https://example.com/entry</id>
                <updated>2024-01-01T10:00:00Z</updated>
                <summary>Entry summary</summary>
            </entry>
        </feed>'''
        
        parsed_feed = await parser.parse_feed(atom_no_ns)
        
        assert parsed_feed.metadata.title == "Atom without namespace"
        assert len(parsed_feed.items) == 1
        assert parsed_feed.items[0].title == "Entry without namespace"
    
    @pytest.mark.asyncio
    async def test_empty_elements(self, parser):
        """Test feed with empty elements."""
        empty_elements_rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Feed with Empty Elements</title>
                <link>https://example.com</link>
                <description>Test description</description>
                <language></language>
                <copyright></copyright>
                
                <item>
                    <title>Item with Empty Elements</title>
                    <link>https://example.com/item</link>
                    <description>Item description</description>
                    <author></author>
                    <pubDate></pubDate>
                    <category></category>
                </item>
            </channel>
        </rss>'''
        
        parsed_feed = await parser.parse_feed(empty_elements_rss)
        
        assert parsed_feed.metadata.title == "Feed with Empty Elements"
        assert parsed_feed.metadata.language == ""
        assert parsed_feed.metadata.copyright == ""
        
        item = parsed_feed.items[0]
        assert item.title == "Item with Empty Elements"
        assert item.author == ""
        assert item.published_at is None