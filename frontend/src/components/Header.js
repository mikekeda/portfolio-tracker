import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  AccountBalance as HoldingsIcon,
  PieChart as AllocationsIcon,
  TrendingUp as PerformanceIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/holdings', label: 'Holdings', icon: <HoldingsIcon /> },
    { path: '/allocations', label: 'Allocations', icon: <AllocationsIcon /> },
    { path: '/performance', label: 'Performance', icon: <PerformanceIcon /> },
  ];

  return (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 0, mr: 4 }}>
          📊 Trading212 Portfolio
        </Typography>

        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              startIcon={item.icon}
              onClick={() => navigate(item.path)}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
