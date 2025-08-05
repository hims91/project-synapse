/**
 * Project Synapse - Dendrites (Feed Poller)
 * Cloudflare Worker for high-frequency RSS/Atom feed polling
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Handle different routes
    switch (url.pathname) {
      case '/health':
        return handleHealth();
      case '/poll':
        return handlePoll(request, env);
      case '/feeds':
        return handleFeeds(request, env);
      default:
        return new Response('Not Found', { status: 404 });
    }
  },

  async scheduled(event, env, ctx) {
    // Handle cron triggers
    console.log('Cron trigger fired:', event.cron);
    
    try {
      await pollFeeds(env);
    } catch (error) {
      console.error('Error in scheduled polling:', error);
    }
  }
};

async function handleHealth() {
  return Response.json({
    status: 'healthy',
    component: 'dendrites',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
}

async function handlePoll(request, env) {
  if (request.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 });
  }
  
  try {
    const result = await pollFeeds(env);
    return Response.json({
      status: 'success',
      feeds_polled: result.feedsPolled,
      items_found: result.itemsFound
    });
  } catch (error) {
    console.error('Error polling feeds:', error);
    return Response.json({
      status: 'error',
      message: error.message
    }, { status: 500 });
  }
}

async function handleFeeds(request, env) {
  if (request.method === 'GET') {
    // List feeds
    const feeds = await env.FEED_CACHE.list();
    return Response.json({
      feeds: feeds.keys.map(key => key.name)
    });
  } else if (request.method === 'POST') {
    // Add new feed
    const { url, priority = 'normal' } = await request.json();
    
    if (!url) {
      return Response.json({
        error: 'URL is required'
      }, { status: 400 });
    }
    
    // Store feed configuration
    await env.FEED_CACHE.put(`feed:${url}`, JSON.stringify({
      url,
      priority,
      added_at: new Date().toISOString(),
      last_polled: null,
      status: 'active'
    }));
    
    return Response.json({
      status: 'success',
      message: 'Feed added successfully'
    });
  }
  
  return new Response('Method not allowed', { status: 405 });
}

async function pollFeeds(env) {
  console.log('Starting feed polling...');
  
  let feedsPolled = 0;
  let itemsFound = 0;
  
  try {
    // Get all feeds from KV storage
    const feedsList = await env.FEED_CACHE.list({ prefix: 'feed:' });
    
    for (const feedKey of feedsList.keys) {
      try {
        const feedData = await env.FEED_CACHE.get(feedKey.name);
        if (!feedData) continue;
        
        const feed = JSON.parse(feedData);
        console.log(`Polling feed: ${feed.url}`);
        
        // Fetch and parse feed
        const response = await fetch(feed.url, {
          headers: {
            'User-Agent': 'Project Synapse Feed Poller 1.0'
          }
        });
        
        if (!response.ok) {
          console.error(`Failed to fetch feed ${feed.url}: ${response.status}`);
          continue;
        }
        
        const feedContent = await response.text();
        const items = await parseFeed(feedContent, feed.url);
        
        if (items.length > 0) {
          // Send items to hub
          await sendItemsToHub(items, env);
          itemsFound += items.length;
          
          // Update last polled time
          feed.last_polled = new Date().toISOString();
          await env.FEED_CACHE.put(feedKey.name, JSON.stringify(feed));
        }
        
        feedsPolled++;
        
      } catch (error) {
        console.error(`Error polling feed ${feedKey.name}:`, error);
      }
    }
    
    console.log(`Polling complete: ${feedsPolled} feeds, ${itemsFound} items`);
    
    return { feedsPolled, itemsFound };
    
  } catch (error) {
    console.error('Error in pollFeeds:', error);
    throw error;
  }
}

async function parseFeed(feedContent, feedUrl) {
  // Simple RSS/Atom parser
  const items = [];
  
  try {
    // Parse XML (simplified - in production use a proper XML parser)
    const itemMatches = feedContent.match(/<item[^>]*>[\s\S]*?<\/item>/gi) || 
                       feedContent.match(/<entry[^>]*>[\s\S]*?<\/entry>/gi) || [];
    
    for (const itemXml of itemMatches.slice(0, 10)) { // Limit to 10 items per feed
      const item = {
        feed_url: feedUrl,
        title: extractXmlContent(itemXml, 'title'),
        link: extractXmlContent(itemXml, 'link') || extractXmlAttribute(itemXml, 'link', 'href'),
        description: extractXmlContent(itemXml, 'description') || extractXmlContent(itemXml, 'summary'),
        pub_date: extractXmlContent(itemXml, 'pubDate') || extractXmlContent(itemXml, 'published'),
        guid: extractXmlContent(itemXml, 'guid') || extractXmlContent(itemXml, 'id'),
        discovered_at: new Date().toISOString()
      };
      
      if (item.title && item.link) {
        items.push(item);
      }
    }
    
  } catch (error) {
    console.error('Error parsing feed:', error);
  }
  
  return items;
}

function extractXmlContent(xml, tag) {
  const regex = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i');
  const match = xml.match(regex);
  return match ? match[1].trim().replace(/<!\[CDATA\[(.*?)\]\]>/g, '$1') : null;
}

function extractXmlAttribute(xml, tag, attribute) {
  const regex = new RegExp(`<${tag}[^>]*${attribute}=["']([^"']*?)["'][^>]*>`, 'i');
  const match = xml.match(regex);
  return match ? match[1] : null;
}

async function sendItemsToHub(items, env) {
  try {
    const hubUrl = env.SYNAPSE_HUB_URL || 'https://synapse-central-hub.onrender.com';
    const apiKey = env.SYNAPSE_API_KEY;
    
    if (!apiKey) {
      console.error('SYNAPSE_API_KEY not configured');
      return;
    }
    
    const response = await fetch(`${hubUrl}/v1/feeds/items`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        component: 'dendrites',
        items: items,
        timestamp: new Date().toISOString()
      })
    });
    
    if (!response.ok) {
      console.error(`Failed to send items to hub: ${response.status}`);
    } else {
      console.log(`Successfully sent ${items.length} items to hub`);
    }
    
  } catch (error) {
    console.error('Error sending items to hub:', error);
  }
}