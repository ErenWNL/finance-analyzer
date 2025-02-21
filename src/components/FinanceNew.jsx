import React, { useState, useEffect } from 'react';
import { Search, FileText } from 'lucide-react';
import NewsCard from './NewsCard';

const FinanceNews = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [originalNewsData, setOriginalNewsData] = useState([]);
  const [displayedNews, setDisplayedNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTickers] = useState(['AAPL', 'GOOGL', 'TSLA']);

  const fetchNewsData = async () => {
    setLoading(true);
    setError(null);
    
    const options = {
      method: 'GET',
      headers: {
        'X-RapidAPI-Key': '13e98f9a29msh138e2d1d19349b7p10a587jsn3d644f15298a',
        'X-RapidAPI-Host': 'yahoo-finance166.p.rapidapi.com',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    };

    try {
      const symbols = selectedTickers.join('%2C');
      const response = await fetch(
        `https://yahoo-finance166.p.rapidapi.com/api/news/list-by-symbol?s=${symbols}&region=US&snippetCount=500`,
        options
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('API Response:', data);

      if (data?.data?.main?.stream) {
        const newsItems = data.data.main.stream;
        console.log('Setting original news items:', newsItems.length);
        setOriginalNewsData(newsItems);
        setDisplayedNews(newsItems);
      } else {
        setOriginalNewsData([]);
        setDisplayedNews([]);
      }
    } catch (error) {
      console.error('Error fetching news data:', error);
      setError('Failed to fetch news data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNewsData();
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    console.log('Searching for:', searchQuery);
    console.log('Original data length:', originalNewsData.length);

    if (!searchQuery.trim()) {
      console.log('Empty search, restoring original data');
      setDisplayedNews(originalNewsData);
      return;
    }

    const query = searchQuery.toLowerCase();
    const filtered = originalNewsData.filter(item => {
      if (!item?.content) return false;
      
      const content = item.content;
      const titleMatch = content.title?.toLowerCase().includes(query);
      const providerMatch = content.provider?.displayName?.toLowerCase().includes(query);
      const tickerMatch = content.finance?.stockTickers?.some(ticker => 
        ticker.symbol?.toLowerCase().includes(query)
      );

      return titleMatch || providerMatch || tickerMatch;
    });

    console.log('Found filtered results:', filtered.length);
    setDisplayedNews(filtered);
  };

  const handleInputChange = (e) => {
    setSearchQuery(e.target.value);
    if (!e.target.value.trim()) {
      setDisplayedNews(originalNewsData);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0B0E] p-8">
      <div className="max-w-[1200px] mx-auto">
        {/* Header */}
        <div className="flex items-start justify-between mb-12">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <FileText className="text-blue-400 w-8 h-8" />
              <h1 className="text-4xl font-bold text-white">Financial News Dashboard</h1>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => window.history.back()}
                className="bg-[#1A1D24] text-gray-300 hover:text-white px-4 py-2 rounded-lg border border-gray-700 hover:border-blue-500 transition-all duration-200 flex items-center gap-2 hover:bg-[#1E2128]"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 19l-7-7m0 0l7-7m-7 7h18"
                  />
                </svg>
                Back to Dashboard
              </button>
            </div>
          </div>
          <div className="text-gray-400">
            Last Updated: {new Date().toLocaleString()}
          </div>
        </div>

        {/* Search Bar */}
        <div className="flex items-center justify-center mb-16">
          <div className="w-full max-w-2xl relative">
            <div className="absolute left-4 top-1/2 -translate-y-1/2">
              <Search className="w-5 h-5 text-blue-400" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={handleInputChange}
              placeholder="Search by title, source, or ticker..."
              className="w-full bg-[#1A1D24] text-white rounded-full pl-12 pr-24 py-3 border border-gray-800 focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={handleSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-full transition-colors duration-200"
            >
              Search
            </button>
          </div>
        </div>

        {/* Content Area */}
        {error ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-red-400">{error}</div>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-blue-400 animate-pulse">Loading news data...</div>
          </div>
        ) : displayedNews.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-[400px]">
            <FileText className="w-16 h-16 text-gray-600 mb-4" />
            <h3 className="text-xl text-gray-400 mb-2">
              {searchQuery.trim() 
                ? `No results found for "${searchQuery}"`
                : 'No news available'}
            </h3>
            {searchQuery.trim() && (
              <button
                onClick={() => {
                  setSearchQuery('');
                  setDisplayedNews(originalNewsData);
                }}
                className="mt-4 text-blue-400 hover:text-blue-300 transition-colors"
              >
                Clear search
              </button>
            )}
            <p className="text-gray-500 mt-2">
              {searchQuery.trim() 
                ? 'Try different search terms'
                : 'Please try again later'}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {displayedNews.map((item, index) => {
              if (!item?.content) return null;
              
              return (
                <NewsCard
                  key={item.id || index}
                  news={{
                    title: item.content.title,
                    thumbnail: item.content.thumbnail,
                    stockTickers: item.content.finance?.stockTickers || [],
                    pubDate: item.content.pubDate,
                    provider: item.content.provider,
                    clickThroughUrl: item.content.clickThroughUrl
                  }}
                />
              );
            })}
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-12 text-gray-500">
          Powered by Yahoo Finance API
        </div>
      </div>
    </div>
  );
};

export default FinanceNews;