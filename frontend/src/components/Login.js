import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const Login = () => {
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const navigate = useNavigate();

  // Load existing token if available
  useEffect(() => {
    try {
      const existingToken = localStorage.getItem('api_token');
      if (existingToken) {
        setToken(existingToken);
      }
    } catch (err) {
      // Handle cases where localStorage is not available (e.g., private browsing)
      console.error('Error reading token from localStorage:', err);
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    const trimmedToken = token.trim();
    if (!trimmedToken) {
      setError('API token is required');
      setIsLoading(false);
      return;
    }

    try {
      // Save token to localStorage (may fail in private browsing mode)
      localStorage.setItem('api_token', trimmedToken);
      
      // Get the redirect path from sessionStorage (if user was redirected here)
      const redirectPath = sessionStorage.getItem('redirectAfterLogin');
      sessionStorage.removeItem('redirectAfterLogin');
      
      // Redirect to the stored path or home
      navigate(redirectPath || '/');
    } catch (err) {
      // Handle localStorage quota exceeded or private browsing mode
      setError('Failed to save token. Please check your browser settings.');
      console.error('Error saving token:', err);
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    try {
      localStorage.removeItem('api_token');
      setToken('');
      setError('');
    } catch (err) {
      console.error('Error clearing token:', err);
      setError('Failed to clear token. Please try again.');
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>Trading212 Portfolio Manager</h1>
        <h2>API Token Configuration</h2>
        <p className="login-description">
          Please enter your API token to access the portfolio data.
        </p>
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="api-token">API Token</label>
            <div className="input-wrapper">
              <input
                id="api-token"
                type={showToken ? 'text' : 'password'}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Enter your API token"
                className={error ? 'error-input' : ''}
                autoFocus
                disabled={isLoading}
              />
              {token && (
                <button
                  type="button"
                  className={`toggle-visibility ${showToken ? 'visible' : ''}`}
                  onClick={() => setShowToken(!showToken)}
                  aria-label={showToken ? 'Hide token' : 'Show token'}
                  aria-pressed={showToken}
                  title={showToken ? 'Hide token' : 'Show token'}
                >
                  <span aria-hidden="true">{showToken ? '👁️' : '👁️‍🗨️'}</span>
                </button>
              )}
            </div>
            {error && <div className="error-message">{error}</div>}
          </div>

          <div className="form-actions">
            <button
              type="submit"
              className="btn-primary"
              disabled={isLoading || !token.trim()}
            >
              {isLoading ? 'Saving...' : 'Save Token'}
            </button>
            {token && (
              <button
                type="button"
                onClick={handleClear}
                className="btn-secondary"
                disabled={isLoading}
              >
                Clear Token
              </button>
            )}
          </div>
        </form>

        <div className="login-footer">
          <p className="help-text">
            Your token is stored locally in your browser and will be included in all API requests.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
