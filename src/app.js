const express = require('express');
const cors = require('cors');

const app = express();

app.use(cors());
app.use(express.json());

// Health Check Endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'Server is healthy and running smoothly',
    timestamp: new Date().toISOString()
  });
});

module.exports = app;