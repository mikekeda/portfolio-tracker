# Trading212 Portfolio Frontend

A React-based frontend for the Trading212 Portfolio Manager.

## Features

- **Portfolio Dashboard**: Overview of portfolio performance and statistics
- **Holdings Table**: Detailed view of all portfolio holdings with sorting and filtering
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Data**: Fetches live data from the backend API

## Technology Stack

- **React 19.1.1**: Latest React with hooks
- **React Router DOM 7.8.2**: Client-side routing
- **@tanstack/react-table 8.21.3**: Advanced table functionality
- **Axios 1.11.0**: HTTP client for API calls
- **Recharts 3.1.2**: Charting library (for future use)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables (optional):
Create a `.env.local` file in the root directory:
```bash
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Environment
REACT_APP_ENV=development

# Feature flags
REACT_APP_ENABLE_ANALYTICS=false
REACT_APP_ENABLE_DEBUG_LOGGING=true
```

3. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend API base URL |
| `REACT_APP_ENV` | `development` | Environment (development/production) |
| `REACT_APP_ENABLE_ANALYTICS` | `false` | Enable analytics tracking |
| `REACT_APP_ENABLE_DEBUG_LOGGING` | `true` | Enable debug logging |

## Project Structure

```
src/
├── components/          # React components
│   ├── Dashboard.js     # Portfolio overview
│   ├── Holdings.js      # Holdings table
│   └── Navigation.js    # Header navigation
├── services/            # API services
│   └── api.js          # Centralized API client
├── config.js           # Application configuration
├── App.js              # Main app component
└── index.js            # Entry point
```

## API Integration

The frontend uses a centralized API service layer (`src/services/api.js`) that provides:

- **Automatic error handling**: Interceptors for request/response errors
- **Request logging**: Debug logging for API calls
- **Timeout handling**: Configurable request timeouts
- **Environment configuration**: Uses environment variables for API URLs

### Available API Methods

```javascript
import { portfolioAPI } from '../services/api';

// Get current holdings
const holdings = await portfolioAPI.getCurrentHoldings();

// Get portfolio summary
const summary = await portfolioAPI.getSummary();

// Get portfolio allocations
const allocations = await portfolioAPI.getAllocations();

// Get portfolio history
const history = await portfolioAPI.getHistory(30); // 30 days
```

## Development

### Available Scripts

- `npm start`: Start development server
- `npm build`: Build for production
- `npm test`: Run tests
- `npm eject`: Eject from Create React App

### Code Style

- Use functional components with hooks
- Follow React best practices
- Use CSS modules for styling
- Implement proper error handling

## Deployment

1. Build the application:
```bash
npm run build
```

2. Deploy the `build/` directory to your web server

3. Configure environment variables for production:
```bash
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_ENV=production
REACT_APP_ENABLE_ANALYTICS=true
REACT_APP_ENABLE_DEBUG_LOGGING=false
```

## Contributing

1. Follow the existing code style
2. Add proper error handling
3. Test your changes
4. Update documentation as needed
