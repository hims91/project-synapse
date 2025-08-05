/**
 * Feed Parser for Cloudflare Workers
 * Lightweight RSS/Atom parser optimized for edge computing
 */

import { XMLParser } from 'fast-xml-parser';

export class FeedParser {
  constructor() {
    this.xmlParser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '@_',
      textNodeName: '#text',
      parseAttributeValue: true,
      trimValues: true,
      parseTrueNumberOnly: false
    });
  }

  /**
   * Parse RSS/Atom feed from XML content
   */
  async parseFeed(xmlContent, feedUrl = '') {
    try {
      // Clean XML content
      const cleanedXml = this.cleanXml(xmlContent);
      
      // Parse XML
      const parsed = this.xmlParser.parse(cleanedXml);
      
      // Detect feed type and parse accordingly
      if (parsed.rss) {
        return this.parseRss(parsed.rss, feedUrl);
      } else if (parsed.feed) {
        return this.parseAtom(parsed.feed, feedUrl);
      } else if (parsed['rdf:RDF']) {
        return this.parseRss10(parsed['rdf:RDF'], feedUrl);
      } else {
        throw new Error('Unsupported feed format');
      }
      
    } catch (error) {
      console.error('Feed parsing error:', error);
      throw new Error(`Failed to parse feed: ${error.message}`);
    }
  }

  /**
   * Clean XML content
   */
  cleanXml(xmlContent) {
    // Remove BOM if present
    if (xmlContent.charCodeAt(0) === 0xFEFF) {
      xmlContent = xmlContent.slice(1);
    }
    
    // Basic cleanup
    return xmlContent
      .trim()
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;amp;/g, '&amp;');
  }

  /**
   * Parse RSS 2.0 feed
   */
  parseRss(rss, feedUrl) {
    const channel = rss.channel;
    if (!channel) {
      throw new Error('Invalid RSS feed: missing channel');
    }

    // Parse metadata
    const metadata = {
      title: this.getText(channel.title),
      link: this.getText(channel.link),
      description: this.getText(channel.description),
      language: this.getText(channel.language),
      lastBuildDate: this.parseDate(channel.lastBuildDate),
      pubDate: this.parseDate(channel.pubDate),
      generator: this.getText(channel.generator),
      feedType: 'rss_2.0'
    };

    // Parse items
    const items = [];
    const channelItems = Array.isArray(channel.item) ? channel.item : [channel.item].filter(Boolean);
    
    for (const item of channelItems) {
      const parsedItem = this.parseRssItem(item, feedUrl);
      if (parsedItem) {
        items.push(parsedItem);
      }
    }

    return {
      metadata,
      items,
      parseTime: new Date().toISOString()
    };
  }

  /**
   * Parse RSS item
   */
  parseRssItem(item, feedUrl) {
    if (!item) return null;

    return {
      title: this.getText(item.title),
      link: this.getText(item.link),
      description: this.getText(item.description),
      content: this.getText(item['content:encoded']) || this.getText(item.content),
      author: this.getText(item.author) || this.getText(item['dc:creator']),
      publishedAt: this.parseDate(item.pubDate),
      guid: this.getText(item.guid),
      categories: this.parseCategories(item.category),
      enclosures: this.parseEnclosures(item.enclosure)
    };
  }

  /**
   * Parse Atom feed
   */
  parseAtom(feed, feedUrl) {
    // Parse metadata
    const metadata = {
      title: this.getText(feed.title),
      link: this.getAtomLink(feed.link),
      description: this.getText(feed.subtitle) || this.getText(feed.summary),
      language: this.getText(feed['@_xml:lang']),
      lastBuildDate: this.parseDate(feed.updated),
      generator: this.getText(feed.generator),
      feedType: 'atom_1.0'
    };

    // Parse entries
    const items = [];
    const entries = Array.isArray(feed.entry) ? feed.entry : [feed.entry].filter(Boolean);
    
    for (const entry of entries) {
      const parsedItem = this.parseAtomEntry(entry, feedUrl);
      if (parsedItem) {
        items.push(parsedItem);
      }
    }

    return {
      metadata,
      items,
      parseTime: new Date().toISOString()
    };
  }

  /**
   * Parse Atom entry
   */
  parseAtomEntry(entry, feedUrl) {
    if (!entry) return null;

    return {
      title: this.getText(entry.title),
      link: this.getAtomLink(entry.link),
      description: this.getText(entry.summary),
      content: this.getText(entry.content),
      author: this.getAtomAuthor(entry.author),
      publishedAt: this.parseDate(entry.published),
      updatedAt: this.parseDate(entry.updated),
      guid: this.getText(entry.id),
      categories: this.parseAtomCategories(entry.category)
    };
  }

  /**
   * Parse RSS 1.0 (RDF) feed
   */
  parseRss10(rdf, feedUrl) {
    const channel = rdf.channel;
    if (!channel) {
      throw new Error('Invalid RSS 1.0 feed: missing channel');
    }

    // Parse metadata
    const metadata = {
      title: this.getText(channel.title),
      link: this.getText(channel.link),
      description: this.getText(channel.description),
      feedType: 'rss_1.0'
    };

    // Parse items
    const items = [];
    const rdfItems = Array.isArray(rdf.item) ? rdf.item : [rdf.item].filter(Boolean);
    
    for (const item of rdfItems) {
      const parsedItem = this.parseRss10Item(item, feedUrl);
      if (parsedItem) {
        items.push(parsedItem);
      }
    }

    return {
      metadata,
      items,
      parseTime: new Date().toISOString()
    };
  }

  /**
   * Parse RSS 1.0 item
   */
  parseRss10Item(item, feedUrl) {
    if (!item) return null;

    return {
      title: this.getText(item.title),
      link: this.getText(item.link),
      description: this.getText(item.description),
      author: this.getText(item['dc:creator']),
      publishedAt: this.parseDate(item['dc:date']),
      guid: this.getText(item['@_rdf:about']) || this.getText(item.link)
    };
  }

  /**
   * Helper methods
   */
  getText(value) {
    if (!value) return '';
    
    if (typeof value === 'string') {
      return value.trim();
    }
    
    if (typeof value === 'object' && value['#text']) {
      return value['#text'].trim();
    }
    
    return String(value).trim();
  }

  parseDate(dateStr) {
    if (!dateStr) return null;
    
    const dateString = this.getText(dateStr);
    if (!dateString) return null;
    
    try {
      const date = new Date(dateString);
      return isNaN(date.getTime()) ? null : date.toISOString();
    } catch {
      return null;
    }
  }

  parseCategories(categories) {
    if (!categories) return [];
    
    const categoryArray = Array.isArray(categories) ? categories : [categories];
    return categoryArray.map(cat => this.getText(cat)).filter(Boolean);
  }

  parseEnclosures(enclosures) {
    if (!enclosures) return [];
    
    const enclosureArray = Array.isArray(enclosures) ? enclosures : [enclosures];
    return enclosureArray.map(enc => ({
      url: enc['@_url'] || '',
      type: enc['@_type'] || '',
      length: enc['@_length'] || ''
    })).filter(enc => enc.url);
  }

  getAtomLink(links) {
    if (!links) return '';
    
    const linkArray = Array.isArray(links) ? links : [links];
    
    // Look for alternate link first
    for (const link of linkArray) {
      if (link['@_rel'] === 'alternate' && link['@_href']) {
        return link['@_href'];
      }
    }
    
    // Fallback to first link with href
    for (const link of linkArray) {
      if (link['@_href']) {
        return link['@_href'];
      }
    }
    
    return '';
  }

  getAtomAuthor(author) {
    if (!author) return '';
    
    if (typeof author === 'string') {
      return author;
    }
    
    if (author.name) {
      return this.getText(author.name);
    }
    
    return '';
  }

  parseAtomCategories(categories) {
    if (!categories) return [];
    
    const categoryArray = Array.isArray(categories) ? categories : [categories];
    return categoryArray.map(cat => cat['@_term'] || '').filter(Boolean);
  }
}