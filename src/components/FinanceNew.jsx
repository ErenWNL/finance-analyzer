import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, Clock, ExternalLink, Newspaper } from 'lucide-react';
import ErrorState from './ErrorState';
import NewsCard from './NewsCard';

const FinanceNews = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchNews = async (query = '') => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        'https://reuters-business-and-financial-news.p.rapidapi.com/market-rics/list-rics-by-asset-and-category/1/1',
        {
          method: 'GET',
          headers: {
            'x-rapidapi-key': '28e6cbb5ecmshdf582854878ed44p1570b9jsn3820887538bd',
            'x-rapidapi-host': 'reuters-business-and-financial-news.p.rapidapi.com'
          }
        }
      );
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setNews(data.data || []);
    } catch (error) {
      console.error('Error fetching news:', error);
      setError('Failed to fetch news. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNews();
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    fetchNews(searchQuery);
  };

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="container mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8 bg-gray-800 p-6 rounded-lg shadow-lg">
          <h1 className="text-4xl font-bold text-white flex items-center gap-3">
            <Newspaper className="text-blue-400 w-8 h-8" />
            Reuters Financial News
          </h1>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="relative max-w-2xl mx-auto">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="w-6 h-6 text-blue-400" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search Reuters financial news..."
              className="w-full bg-gray-800 text-white text-lg rounded-xl pl-14 pr-32 py-4 border-2 border-gray-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 shadow-lg placeholder-gray-400"
            />
            <div className="absolute inset-y-0 right-0 flex items-center pr-2">
              <button
                type="submit"
                className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg shadow-md transition-colors duration-200 flex items-center gap-2 mr-2"
              >
                <Search className="w-5 h-5" />
                <span className="font-medium">Search</span>
              </button>
            </div>
          </div>
        </form>

        {/* Error State */}
        {error && <ErrorState message={error} />}

        {/* News Grid */}
        {!error && loading ? (
          <div className="flex items-center justify-center p-12">
            <div className="text-xl text-blue-400 font-medium animate-pulse">
              Loading Reuters news...
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {news.map((item) => (
              <NewsCard key={item.id} news={{
                id: item.id,
                title: item.title || 'Reuters Financial Update',
                description: item.description || item.summary || 'Latest financial market updates from Reuters',
                image: item.image_url || '/placeholder-news.jpg',
                url: item.url || '#',
                source: 'Reuters',
                timestamp: item.published_at ? new Date(item.published_at).toLocaleString() : 'Recent'
              }} />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && (!news || news.length === 0) && (
          <div className="text-center py-12">
            <Newspaper className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl text-gray-400 font-medium">No news articles available</h3>
            <p className="text-gray-500 mt-2">Please try again later</p>
          </div>
        )}

        {/* Reuters Attribution */}
        <div className="text-center mt-8 text-gray-500">
          <p>Powered by Reuters Financial News</p>
        </div>
      </div>
    </div>
  );
};

export default FinanceNews;