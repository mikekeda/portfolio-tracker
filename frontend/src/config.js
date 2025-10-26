// Application configuration
const config = {
  // API Configuration

  // auto-detect based on environment
  API_BASE_URL: (() => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    return hostname === 'localhost'
      ? 'http://localhost:8000'
      : `${protocol}//${hostname}`;
  })(),

  // API timeouts
  API_TIMEOUT: 30000,
};

export default config;
