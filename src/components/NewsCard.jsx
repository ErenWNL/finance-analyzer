import React from 'react';
import { Clock, ExternalLink, Globe } from 'lucide-react';

const NewsCard = ({ news }) => {
  if (!news) return null;

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return 'Date unavailable';
    }
  };

  // Create a default placeholder image URL
  const placeholderImage = '/api/placeholder/800/400';
  
  // Safely get the thumbnail URL
  const thumbnailUrl = news?.thumbnail?.resolutions?.[0]?.url || placeholderImage;

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden hover:shadow-2xl transition-shadow duration-300 border border-gray-700">
      <div className="relative h-48 overflow-hidden">
        <img
          src={thumbnailUrl}
          alt={news.title || 'News thumbnail'}
          className="w-full h-full object-cover transform hover:scale-105 transition-transform duration-300"
          onError={(e) => {
            e.target.src = placeholderImage;
          }}
        />
      </div>
      <div className="p-6">
        <h3 className="text-xl font-semibold text-white mb-3 line-clamp-2 hover:text-blue-400 transition-colors">
          {news.title || 'No title available'}
        </h3>
        {news.stockTickers && news.stockTickers.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {news.stockTickers.map((ticker, index) => (
              <span 
                key={index} 
                className="bg-blue-500 bg-opacity-20 text-blue-400 px-3 py-1 rounded-full text-sm"
              >
                {ticker.symbol}
              </span>
            ))}
          </div>
        )}
        <div className="flex items-center justify-between text-sm mt-4 pt-4 border-t border-gray-700">
          <div className="flex items-center text-gray-400">
            <Clock size={18} className="mr-2" />
            <span>{formatDate(news.pubDate)}</span>
          </div>
          <div className="flex items-center">
            <Globe size={18} className="mr-2 text-gray-400" />
            <span className="text-gray-400 mr-4">
              {news.provider?.displayName || 'Unknown Source'}
            </span>
            <a
              href={news.clickThroughUrl?.url || '#'}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-blue-400 hover:text-blue-300 font-medium transition-colors"
            >
              Read more
              <ExternalLink size={18} className="ml-1" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewsCard;