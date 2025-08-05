"""
Project Synapse - URL Resolver Utility
The definitive URL resolver for handling redirector URLs like Google News.

This module uses a hybrid approach:
1. For RSS/Atom URLs: Uses base64 decoding (reliable and fast)
2. For query param URLs: Extracts from ?url= parameter
3. Fallback: Uses HTTP redirect following when direct extraction fails
"""
import httpx
import structlog
import base64
import re
from urllib.parse import unquote, urlparse, parse_qs
from typing import Optional

logger = structlog.get_logger(__name__)

# Async HTTP client
client = httpx.AsyncClient(
    follow_redirects=True,
    timeout=httpx.Timeout(5.0),
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)


def extract_url_from_google_news(url: str) -> Optional[str]:
    """Attempts to extract the real article URL from a Google News redirector URL."""
    # Case 1: Query param ?url=
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "url" in qs:
        return qs["url"][0]
    
    # Case 2: RSS article (often ends with encoded real URL)
    if "/rss/articles/" in url:
        try:
            # Try to extract a final URL from the path using regex
            match = re.search(r"\/rss\/articles\/([^?]*)", url)
            if match:
                encoded_part = match.group(1)
                
                # Handle CBM prefix - this is Google's encoding format
                if encoded_part.startswith('CBM'):
                    try:
                        # Add padding for base64 decoding
                        padded = encoded_part + '=' * (4 - len(encoded_part) % 4)
                        decoded_bytes = base64.urlsafe_b64decode(padded)
                        decoded_str = decoded_bytes.decode('utf-8', errors='ignore')
                        
                        # Look for HTTP URLs in the decoded content
                        url_match = re.search(r'https?://[^\s\x00-\x1f\x7f-\x9f]+', decoded_str)
                        if url_match:
                            extracted_url = url_match.group(0)
                            # Clean up any trailing non-URL characters
                            extracted_url = re.sub(r'[^\w\-._~:/?#[\]@!$&\'()*+,;=%]+$', '', extracted_url)
                            return extracted_url
                    except Exception:
                        pass
                        
                # Fallback: try direct base64 decode
                try:
                    decoded_url = base64.urlsafe_b64decode(encoded_part + '==').decode("utf-8")
                    if decoded_url.startswith("http"):
                        return decoded_url
                except Exception:
                    pass
                    
        except Exception:
            pass  # fallback
    
    return None


async def resolve_final_url(url: str) -> str:
    """Resolve a Google News or redirector URL to its final destination."""
    if not is_redirector_url(url):
        return url
    
    logger.info("Attempting to resolve", original_url=url)
    
    # Try to extract directly
    extracted = extract_url_from_google_news(url)
    if extracted:
        logger.info("Extracted final URL without redirect", original_url=url, final_url=extracted)
        return extracted
    
    # Fallback to redirect-based resolution
    try:
        response = await client.get(url)
        return str(response.url)
    except httpx.HTTPError as e:
        logger.warning("Failed to resolve via HTTP", url=url, error=str(e))
        return url


async def resolve_multiple_urls(urls: list[str]) -> dict[str, str]:
    import asyncio
    tasks = [resolve_final_url(url) for url in urls]
    resolved = await asyncio.gather(*tasks, return_exceptions=True)
    return {url: (res if isinstance(res, str) else url) 
            for url, res in zip(urls, resolved)}


def is_redirector_url(url: str) -> bool:
    redirector_domains = [
        "news.google.com",
        "t.co",
        "bit.ly", 
        "tinyurl.com",
        "ow.ly",
    ]
    return any(domain in url for domain in redirector_domains)


async def cleanup_client():
    await client.aclose()


# Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        test_urls = [
            "https://news.google.com/rss/articles/CBMiYWh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3dvcmxkL2FzaWEtcGFjaWZpYy9qYXBhbi10by1yZWxlYXNlLW1vcmUtd2F0ZXItZnJvbS1mdWt1c2hpbWEtcGxhbnQtMjAyMy0xMi0xNS_SAQA?oc=5",
            "https://news.google.com/read/CBMiowJBVV85cUxPcHhpODBMVFJINkRHRk5pamRNY3BDVHZQS0MzekZucmJBNklYRlM3bXRaWmhHWjhod2tMUkhJSnUzVjc3Zll5OEszY0FETnU5V2s5SV9LRjVhaUpXUENUU1JlZmNHcTl2ZlEwTVhJa2FCdWVWSlhJSkxZZ0pFRU1fUEswV25wTV9uX21oUURXaFJXVldzaEg4bXZTcWpuWEVKXzYtNGN6anJWUE11TGUxQk9Cd1RmV0dnRmEzUXc5akl2dnJYQ0dNZnNucEZUcDVqQlpieTFlTlc5RVFGVUc0cmw3YjM4US1Pd0JlQXFQUWdySFlEdi1rVURDVnJWNl96bG8tY082X3NSaFBFRkJPS2pxNmVaTS0tOVlfSW01WE1McW8?hl=en-IN&gl=IN&ceid=IN%3Aen",
            "https://www.reuters.com/world/asia-pacific/japan-to-release-more-water-from-fukushima-plant-2023-12-15"
        ]
        
        for url in test_urls:
            print(f"\n🔗 Original: {url}")
            resolved = await resolve_final_url(url)
            print(f"➡️  Resolved: {resolved}")
        
        await cleanup_client()
    
    asyncio.run(test())