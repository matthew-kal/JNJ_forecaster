import React, { useEffect, useState } from 'react';
import './App.css';
import { FaLinkedin, FaGithub } from "react-icons/fa";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

function App() {
  const [stockData, setStockData] = useState([]);
  const [yAxisDomain, setYAxisDomain] = useState([0, 0]);

  useEffect(() => {
    fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ticker: 'JNJ' }),
    })
    .then(response => response.json())
    .then(data => {
      console.log("Data from backend:", data); // Log the response data
      if (data.results && data.results.length > 0) {
        const results = data.results;
        const newStockData = results.map((value, index) => ({
          name: index < 5 ? `Lookback ${index + 1}` : `Prediction`,
          value: value,
        }));
        setStockData(newStockData);

        // Calculate the Y-axis domain
        const values = results;
        const minValue = Math.floor(Math.min(...values));
        const maxValue = Math.ceil(Math.max(...values));
        setYAxisDomain([minValue, maxValue]);
      } else {
        console.error("No results found in the response data.");
      }
    })
    .catch(error => {
      console.error("Error fetching data:", error);
    });
  }, []);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="label">{label}</p>
          <p className="intro">Value: ${payload[0].value.toFixed(2)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="App">
      <nav className="navbar">
        <ul>
          <li>
            <a> Johnson & Johnson Forecasting Engine. Powered by Machine Learning.</a>
          </li>
        </ul>
      </nav>

      <ResponsiveContainer className='Chart' width="100%" height={400}>
        <LineChart
          data={stockData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={yAxisDomain} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#3b82f6" />
        </LineChart>
      </ResponsiveContainer>

      <div className='sticky'> 
        <nav className="navbar">
          <ul>
            <li className='footer'>
              <a>Built by Kal</a>
              <a href="https://www.linkedin.com/in/matthew-kal/"> <FaLinkedin /> </a>
              <a href="https://github.com/matthew-kal/JNJ_forecaster"> <FaGithub /> </a>
            </li>
          </ul>
        </nav>
      </div>
    </div>
  );
}

export default App;
