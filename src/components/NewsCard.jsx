import React from 'react';
import { Clock, ExternalLink, Globe } from 'lucide-react';

const NewsCard = ({ news }) => {
  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden hover:shadow-2xl transition-shadow duration-300 border border-gray-700">
      {news.image && (
        <div className="relative h-48 overflow-hidden">
          <img
            src={news.image}
            alt={news.title}
            className="w-full h-full object-cover transform hover:scale-105 transition-transform duration-300"
          />
        </div>
      )}
      <div className="p-6">
        <h3 className="text-xl font-semibold text-white mb-3 line-clamp-2 hover:text-blue-400 transition-colors">
          {news.title}
        </h3>
        <p className="text-gray-300 text-base mb-4 line-clamp-3">
          {news.description}
        </p>
        <div className="flex items-center justify-between text-sm mt-4 pt-4 border-t border-gray-700">
          <div className="flex items-center text-gray-400">
            <Clock size={18} className="mr-2" />
            <span>{news.timestamp}</span>
          </div>
          <div className="flex items-center">
            <Globe size={18} className="mr-2 text-gray-400" />
            <span className="text-gray-400 mr-4">{news.source}</span>
            <a
              href={news.url}
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