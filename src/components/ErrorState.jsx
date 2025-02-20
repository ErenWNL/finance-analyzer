import React from 'react';
import { AlertTriangle } from 'lucide-react';

const ErrorState = ({ message }) => {
  return (
    <div className="text-center py-12 bg-gray-800 rounded-lg">
      <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
      <h3 className="text-xl text-gray-200 font-medium">Unable to load news</h3>
      <p className="text-gray-400 mt-2">{message}</p>
      <button 
        onClick={() => window.location.reload()} 
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
      >
        Try Again
      </button>
    </div>
  );
};

export default ErrorState;