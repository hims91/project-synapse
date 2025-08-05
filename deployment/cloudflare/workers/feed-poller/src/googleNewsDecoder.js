/**
 * Google News URL Decoder for Cloudflare Workers
 * Decodes Google News redirect URLs to extract original source URLs
 */

export class GoogleNewsDecoder {
  /**
   * Decode Google News URL to get original source URL
   */
  decodeUrl(googleNewsUrl) {
    if (!googleNewsUrl || !googleNewsUrl.includes('news.google.com')) {
      return null;
    }

    try {
      // Handle different Google News URL formats
      if (googleNewsUrl.includes('/rss/articles/')) {
        return this.decodeRssArticleUrl(googleNewsUrl);
      } else if (googleNewsUrl.includes('/read/')) {
        return this.decodeReadUrl(googleNewsUrl);
      }
      
      return null;
      
    } catch (error) {
      console.error('Google News URL decoding error:', error);
      return null;
    }
  }

  /**
   * Decode RSS article format URLs
   */
  decodeRssArticleUrl(url) {
    try {
      const urlObj = new URL(url);
      const pathSegments = urlObj.pathname.split('/');
      
      // Find the encoded string (usually last segment)
      let encodedString = null;
      
      if (pathSegments.length >= 4 && pathSegments[1] === 'rss' && pathSegments[2] === 'articles') {
        encodedString = pathSegments[3];
      } else if (pathSegments.length >= 2) {
        encodedString = pathSegments[pathSegments.length - 1];
      }
      
      if (!encodedString) {
        return null;
      }
      
      // Base64 decode
      const decoded = this.base64Decode(encodedString);
      if (!decoded) {
        return null;
      }
      
      // Extract URL from decoded bytes
      return this.extractUrlFromBytes(decoded);
      
    } catch (error) {
      console.error('RSS article URL decoding error:', error);
      return null;
    }
  }

  /**
   * Decode read format URLs (experimental)
   */
  decodeReadUrl(url) {
    try {
      if (!url.includes('/read/CBM')) {
        return null;
      }
      
      const match = url.match(/\/read\/(CBMi[a-zA-Z0-9_-]+)/);
      if (!match) {
        return null;
      }
      
      const encodedString = decodeURIComponent(match[1]);
      
      // Base64 decode
      const decoded = this.base64Decode(encodedString);
      if (!decoded) {
        return null;
      }
      
      // Try multiple strategies to extract URL
      return this.extractUrlFromBytes(decoded) || this.extractDomainFromBytes(decoded);
      
    } catch (error) {
      console.error('Read URL decoding error:', error);
      return null;
    }
  }

  /**
   * Base64 decode with padding correction
   */
  base64Decode(encodedString) {
    try {
      // Add padding if needed
      const padding = '='.repeat((4 - (encodedString.length % 4)) % 4);
      const paddedString = encodedString + padding;
      
      // Decode base64
      const binaryString = atob(paddedString);
      
      // Convert to Uint8Array
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      return bytes;
      
    } catch (error) {
      console.error('Base64 decode error:', error);
      return null;
    }
  }

  /**
   * Extract URL from decoded bytes
   */
  extractUrlFromBytes(bytes) {
    try {
      // Convert bytes to string (with error handling)
      let text = '';
      for (let i = 0; i < bytes.length; i++) {
        const byte = bytes[i];
        if (byte >= 32 && byte <= 126) { // Printable ASCII
          text += String.fromCharCode(byte);
        } else if (byte === 0) {
          break; // Stop at null terminator
        }
      }
      
      // Look for HTTP/HTTPS URLs
      const urlRegex = /https?:\/\/[a-zA-Z0-9\-._~:/?#[\]@!&'()*+,;=%]+/g;
      const matches = text.match(urlRegex);
      
      if (matches && matches.length > 0) {
        // Return the longest URL (usually the article URL)
        return matches.reduce((longest, current) => 
          current.length > longest.length ? current : longest
        );
      }
      
      return null;
      
    } catch (error) {
      console.error('URL extraction error:', error);
      return null;
    }
  }

  /**
   * Extract domain from bytes and construct URL (fallback method)
   */
  extractDomainFromBytes(bytes) {
    try {
      // Convert bytes to string
      let text = '';
      for (let i = 0; i < bytes.length; i++) {
        const byte = bytes[i];
        if (byte >= 32 && byte <= 126) {
          text += String.fromCharCode(byte);
        }
      }
      
      // Look for domain patterns
      const domainRegex = /([a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|co\.uk|co\.in|in|today|news|times)[a-zA-Z0-9/._-]*)/g;
      const matches = text.match(domainRegex);
      
      if (matches && matches.length > 0) {
        // Return the longest domain pattern as HTTPS URL
        const longestDomain = matches.reduce((longest, current) => 
          current.length > longest.length ? current : longest
        );
        
        if (longestDomain.length >= 10) {
          return `https://${longestDomain}`;
        }
      }
      
      return null;
      
    } catch (error) {
      console.error('Domain extraction error:', error);
      return null;
    }
  }

  /**
   * Test the decoder with sample URLs
   */
  static test() {
    const decoder = new GoogleNewsDecoder();
    
    // Test RSS format
    const rssUrl = "https://news.google.com/rss/articles/CBMiYmh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3dvcmxkL2FzaWEtcGFjaWZpYy9qYXBhbi10by1yZWxlYXNlLW1vcmUtd2F0ZXItZnJvbS1mdWt1c2hpbWEtcGxhbnQtMjAyMy0xMi0xNS_SAQA?oc=5";
    
    console.log('Testing RSS format:');
    console.log('Original:', rssUrl);
    console.log('Decoded:', decoder.decodeUrl(rssUrl));
    
    // Test read format
    const readUrl = "https://news.google.com/read/CBMiowJBVV95cUxPcHhpODBMVFJINkRHRk5pamRNY3BDVHZQS0MzekZucmJBNklYRlM3bXRaWmhHWjhod2tMUkhJSnUzVjc3Zll5OEszY0FETnU5V2s5SV9LRjVhaUpXUENUU1JlZmNHcTl2ZlEwTVhJa2FCdWVWSlhJSkxZZ0pFRU1fUEswV25wTV9uX21oUURXaFJXVldzaEg4bXZTcWpuWEVKXzYtNGN6anJWUE11TGUxQk9Cd1RmV0dnRmEzUXc5akl2dnJYQ0dNZnNucEZUcDVqQlpieTFlTlc5RVFGVUc0cmw3YjM4US1Pd0JlQXFQUWdySFlEdi1rVURDVnJWNl96bG8tY082X3NSaFBFRkJPS2pxNmVaTS0tOVlfSW01WE1McW8?hl=en-IN&gl=IN&ceid=IN%3Aen";
    
    console.log('Testing read format:');
    console.log('Original:', readUrl);
    console.log('Decoded:', decoder.decodeUrl(readUrl));
  }
}