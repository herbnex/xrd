import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  Box,
  IconButton,
  Divider,
  Button,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  TextField,
  Container,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Menu as MenuIcon, UploadFile as UploadIcon } from '@mui/icons-material';
import { Responsive, WidthProvider } from 'react-grid-layout';
import {
  ResponsiveContainer,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line
} from 'recharts';

const ResponsiveGridLayout = WidthProvider(Responsive);

/** Helper to display multi-paragraph text. */
function ReportDisplay({ text }) {
  if (!text) return null;
  return text.split(/\n\s*\n/).map((p, idx) => (
    <Typography key={idx} variant="body2" paragraph sx={{ whiteSpace: 'pre-wrap' }}>
      {p}
    </Typography>
  ));
}

/** Combine multiple numeric steps + fitted data into multi-line chart. */
function unifyDataForMultiLine(analysisResult) {
  // We'll store multiple lines: raw, calibrated, bg, smoothed, stripped, fitted
  if (!analysisResult) return [];

  const map = new Map();

  function addSeries(arr, key) {
    arr.forEach(d => {
      const x = d.two_theta;
      if (!map.has(x)) {
        map.set(x, { two_theta: x });
      }
      map.get(x)[key] = d.intensity;
    });
  }

  const pd = analysisResult.parsedData || [];
  const cd = analysisResult.calibratedData || [];
  const bg = analysisResult.bgCorrectedData || [];
  const sm = analysisResult.smoothedData || [];
  const ka = analysisResult.strippedData || [];
  const fp = analysisResult.fittedPeaks || [];

  addSeries(pd, 'Raw');
  addSeries(cd, 'Calib');
  addSeries(bg, 'BG');
  addSeries(sm, 'Smooth');
  addSeries(ka, 'Kalpha');
  fp.forEach(f => {
    const x = f.two_theta;
    if (!map.has(x)) {
      map.set(x, { two_theta: x });
    }
    map.get(x).Fitted = f.intensity;
  });

  return Array.from(map.values()).sort((a,b)=>a.two_theta - b.two_theta);
}

const LEFT_DRAWER_WIDTH = 280;
const RIGHT_DRAWER_WIDTH = 320;

const RootBox = styled('div')({
  display: 'flex',
  minHeight: '100vh',
  backgroundColor: '#f9f9f9'
});

const MainContent = styled('main')(({ theme, leftOpen, rightOpen }) => ({
  flexGrow: 1,
  marginTop: theme.spacing(8),
  marginLeft: leftOpen ? LEFT_DRAWER_WIDTH : 0,
  marginRight: rightOpen ? RIGHT_DRAWER_WIDTH : 0,
  transition: theme.transitions.create(['margin'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen
  })
}));

export default function App() {
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(true);
  const [rightDrawerOpen, setRightDrawerOpen] = useState(true);

  // File states
  const [singleFile, setSingleFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Extra states
  const [multiFiles, setMultiFiles] = useState([]);
  const [clusterResult, setClusterResult] = useState(null);
  const [simulationText, setSimulationText] = useState('');
  const [simulationResult, setSimulationResult] = useState(null);

  // UI
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // --------------- Numeric Settings ---------------
  // If left blank or minimal, GPT will fill them in via recommend_numeric_params
  const [bgMethod, setBgMethod] = useState('');
  const [waveletName, setWaveletName] = useState('');
  const [polyOrder, setPolyOrder] = useState('');
  const [smoothingMethod, setSmoothingMethod] = useState('');
  const [kalphaFraction, setKalphaFraction] = useState('');
  const [calibrationOffset, setCalibrationOffset] = useState('');
  const [enableIterRefine, setEnableIterRefine] = useState(true);

  // For multi-line chart
  const chartData = unifyDataForMultiLine(analysisResult);

  const layouts = {
    lg: [
      { i: 'diffPattern', x: 0, y: 0, w: 8, h: 12 },
      { i: 'finalReport', x: 8, y: 0, w: 4, h: 12 }
    ],
    md: [
      { i: 'diffPattern', x: 0, y: 0, w: 6, h: 12 },
      { i: 'finalReport', x: 6, y: 0, w: 6, h: 12 }
    ],
    sm: [
      { i: 'diffPattern', x: 0, y: 0, w: 6, h: 12 },
      { i: 'finalReport', x: 0, y: 12, w: 6, h: 12 }
    ]
  };

  // For local dev, you might use 'http://localhost:8080', etc.
  const API_BASE = 'https://xrd-backend-enuq.onrender.com';

  const toggleLeftDrawer = () => setLeftDrawerOpen(!leftDrawerOpen);
  const toggleRightDrawer = () => setRightDrawerOpen(!rightDrawerOpen);

  /******************************************************
   * Single-file
   ******************************************************/
  const handleSingleFileChange = e => {
    setSingleFile(e.target.files[0]);
    setAnalysisResult(null);
    setClusterResult(null);
    setSimulationResult(null);
    setErrorMessage('');
  };

  const runAnalyze = async () => {
    if (!singleFile) return;
    setIsLoading(true);
    setAnalysisResult(null);
    setErrorMessage('');
    try {
      const formData = new FormData();
      formData.append('xrdFile', singleFile);

      // pass settings as a JSON string. GPT will fill missing values.
      const numericSettings = {};
      if(bgMethod) numericSettings.bgMethod = bgMethod;
      if(waveletName) numericSettings.wavelet = waveletName;
      if(polyOrder) numericSettings.polyOrder = parseInt(polyOrder);
      if(smoothingMethod) numericSettings.smoothingMethod = smoothingMethod;
      if(kalphaFraction) numericSettings.kalphaFraction = parseFloat(kalphaFraction);
      if(calibrationOffset) numericSettings.calibrationOffset = parseFloat(calibrationOffset);
      numericSettings.enableIterativeRefinement = enableIterRefine;

      formData.append('settings', JSON.stringify(numericSettings));

      const resp = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) throw new Error(`Analyze error: ${resp.statusText}`);
      const data = await resp.json();
      setAnalysisResult(data);
    } catch (err) {
      console.error(err);
      setErrorMessage('Analysis failed: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /******************************************************
   * Multi-file cluster
   ******************************************************/
  const handleMultiFileChange = e => {
    const files = Array.from(e.target.files);
    setMultiFiles(files);
    setAnalysisResult(null);
    setClusterResult(null);
    setSimulationResult(null);
    setErrorMessage('');
  };

  const runCluster = async () => {
    if (!multiFiles.length) return;
    setIsLoading(true);
    setAnalysisResult(null);
    setClusterResult(null);
    setSimulationResult(null);
    setErrorMessage('');
    try {
      const formData = new FormData();
      multiFiles.forEach(f => formData.append('clusterFiles', f));
      const resp = await fetch(`${API_BASE}/api/cluster`, {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) throw new Error(`Cluster error: ${resp.statusText}`);
      const data = await resp.json();
      setClusterResult(data);
    } catch(err) {
      console.error(err);
      setErrorMessage('Cluster failed: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /******************************************************
   * Simulation
   ******************************************************/
  const runSimulate = async () => {
    if (!simulationText.trim()) return;
    setIsLoading(true);
    setAnalysisResult(null);
    setClusterResult(null);
    setSimulationResult(null);
    setErrorMessage('');
    try {
      const resp = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ structure: simulationText })
      });
      if(!resp.ok) throw new Error(`Simulation error: ${resp.statusText}`);
      const data = await resp.json();
      setSimulationResult(data);
    } catch(err) {
      console.error(err);
      setErrorMessage('Simulation error: '+err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /******************************************************
   * Render
   ******************************************************/
  const chartDataExists = chartData.length>0;
  return (
    <RootBox>
      <AppBar position="fixed" sx={{ zIndex: theme=> theme.zIndex.drawer +1 }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={toggleLeftDrawer} sx={{ mr:2 }}>
            <MenuIcon/>
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow:1 }}>
            XRD Dashboard (Advanced Numeric + GPT)
          </Typography>
          <Button color="inherit" onClick={toggleRightDrawer}>
            {rightDrawerOpen ? 'Hide Right Pane' : 'Show Right Pane'}
          </Button>
        </Toolbar>
      </AppBar>

      {/* LEFT DRAWER */}
      <Drawer
        variant="temporary"
        open={leftDrawerOpen}
        onClose={toggleLeftDrawer}
        sx={{
          '& .MuiDrawer-paper': {
            width:LEFT_DRAWER_WIDTH,
            boxSizing:'border-box',
            backgroundColor:'#f5f5f5',
            p:2,
            pt:8
          }
        }}
      >
        <Typography variant="h6" gutterBottom color="primary">
          XRD Tools
        </Typography>
        <Divider sx={{ mb:2 }} />

        {/* Single-file analysis */}
        <Typography variant="subtitle1" sx={{ mb:1 }}>Single File Analysis</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon/>} sx={{ mb:1 }}>
          Select File
          <input hidden type="file" accept=".xy,.txt" onChange={handleSingleFileChange}/>
        </Button>
        {singleFile && <Typography variant="body2">{singleFile.name}</Typography>}
        <Button variant="contained" onClick={runAnalyze} disabled={!singleFile || isLoading} fullWidth sx={{ mb:2 }}>
          Analyze
        </Button>

        {/* Multi-file cluster */}
        <Typography variant="subtitle1" sx={{ mb:1 }}>Multi-file Cluster</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon/>} sx={{ mb:1 }}>
          Select Files
          <input hidden type="file" multiple accept=".xy,.txt" onChange={handleMultiFileChange}/>
        </Button>
        {multiFiles.length>0 && (
          <Typography variant="body2">
            {multiFiles.map(f=>f.name).join(', ')}
          </Typography>
        )}
        <Button variant="contained" onClick={runCluster} disabled={!multiFiles.length || isLoading} fullWidth>
          Cluster
        </Button>

        <Divider sx={{ my:2 }}/>

        {/* Simulation */}
        <Typography variant="subtitle1" sx={{ mb:1 }}>Simulation</Typography>
        <TextField
          multiline
          rows={4}
          variant="outlined"
          placeholder="Enter structure..."
          value={simulationText}
          onChange={e=> setSimulationText(e.target.value)}
          sx={{ width:'100%', mb:1 }}
        />
        <Button variant="contained" onClick={runSimulate} disabled={!simulationText.trim() || isLoading} fullWidth>
          Simulate
        </Button>

        <Divider sx={{ my:2 }}/>

        {/* Numeric Settings: GPT will fill missing fields */}
        <Typography variant="h6" color="primary" sx={{ mb:1 }}>
          Numeric Settings
        </Typography>

        <Typography variant="caption">If left blank, GPT decides.</Typography>
        <Divider sx={{ mb:1 }}/>
        <Typography variant="subtitle2">Background Method:</Typography>
        <FormControl fullWidth size="small" sx={{ mb:1 }}>
          <InputLabel>bgMethod</InputLabel>
          <Select
            value={bgMethod}
            label="bgMethod"
            onChange={e=>setBgMethod(e.target.value)}
          >
            <MenuItem value="">Auto (GPT)</MenuItem>
            <MenuItem value="iterative_poly">Iterative Polynomial</MenuItem>
            <MenuItem value="wavelet">Wavelet</MenuItem>
          </Select>
        </FormControl>

        <TextField
          label="Wavelet"
          size="small"
          variant="outlined"
          value={waveletName}
          onChange={e=>setWaveletName(e.target.value)}
          sx={{ mb:1 }}
        />

        <TextField
          label="Poly Order"
          size="small"
          type="number"
          variant="outlined"
          value={polyOrder}
          onChange={e=>setPolyOrder(e.target.value)}
          sx={{ mb:1 }}
        />

        <Typography variant="subtitle2">Smoothing Method:</Typography>
        <FormControl fullWidth size="small" sx={{ mb:1 }}>
          <InputLabel>smoothingMethod</InputLabel>
          <Select
            value={smoothingMethod}
            label="smoothingMethod"
            onChange={e=>setSmoothingMethod(e.target.value)}
          >
            <MenuItem value="">Auto (GPT)</MenuItem>
            <MenuItem value="savitzky_golay">Savitzky-Golay</MenuItem>
            <MenuItem value="average">Moving Average</MenuItem>
          </Select>
        </FormControl>

        <TextField
          label="Kα2 fraction"
          type="number"
          size="small"
          variant="outlined"
          value={kalphaFraction}
          onChange={e=>setKalphaFraction(e.target.value)}
          sx={{ mb:1 }}
        />

        <TextField
          label="Calibration Offset (deg)"
          type="number"
          size="small"
          variant="outlined"
          value={calibrationOffset}
          onChange={e=>setCalibrationOffset(e.target.value)}
          sx={{ mb:1 }}
        />

        <FormControlLabel
          control={<Switch checked={enableIterRefine} onChange={e=>setEnableIterRefine(e.target.checked)}/>}
          label="Enable Iterative Refinement"
        />
      </Drawer>

      {/* RIGHT DRAWER */}
      <Drawer
        variant="persistent"
        anchor="right"
        open={rightDrawerOpen}
        sx={{
          '& .MuiDrawer-paper': {
            width: RIGHT_DRAWER_WIDTH,
            boxSizing:'border-box',
            backgroundColor:'#f5f5f5',
            p:2,
            pt:8
          }
        }}
      >
        <Typography variant="h6" color="primary" sx={{ mb:2 }}>
          Detailed Results
        </Typography>
        {!analysisResult && (
          <Typography variant="body2" color="text.secondary">
            No analysis data yet.
          </Typography>
        )}
        {analysisResult && (
          <Box sx={{ overflowY:'auto', maxHeight:'80vh' }}>
            <Typography variant="subtitle1">R-Factors</Typography>
            <Typography variant="body2">Rwp: {analysisResult.Rwp?.toFixed(4)}</Typography>
            <Typography variant="body2" sx={{ mb:2 }}>Rp: {analysisResult.Rp?.toFixed(4)}</Typography>
            <Divider sx={{ mb:2 }}/>

            {analysisResult.fittedPeaks?.length > 0 && (
              <>
                <Typography variant="subtitle2" sx={{ mt:1 }}>Fitted Peaks</Typography>
                <Box sx={{ maxHeight:100, overflowY:'auto', mb:1 }}>
                  <table style={{ width:'100%', fontSize:'0.8rem' }}>
                    <thead>
                      <tr><th>2θ</th><th>Intensity</th><th>FWHM</th></tr>
                    </thead>
                    <tbody>
                      {analysisResult.fittedPeaks.map((fp, i) => (
                        <tr key={i}>
                          <td>{fp.two_theta.toFixed(3)}</td>
                          <td>{fp.intensity.toFixed(2)}</td>
                          <td>{fp.fwhm.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
                <Divider sx={{ mb:1 }}/>
              </>
            )}

            {analysisResult.phases?.length > 0 && (
              <>
                <Typography variant="subtitle2">Phases</Typography>
                {analysisResult.phases.map((ph, i)=>(
                  <Typography key={i} variant="body2">
                    {ph.phase_name} (conf={ph.confidence.toFixed(2)})
                  </Typography>
                ))}
                <Divider sx={{ my:1 }}/>
              </>
            )}

            {analysisResult.quantResults?.length > 0 && (
              <>
                <Typography variant="subtitle2">Quant</Typography>
                <Box sx={{ maxHeight:100, overflowY:'auto' }}>
                  <table style={{ width:'100%', fontSize:'0.8rem' }}>
                    <thead>
                      <tr><th>Phase</th><th>wt%</th><th>Lattice</th><th>Size</th><th>Conf</th></tr>
                    </thead>
                    <tbody>
                      {analysisResult.quantResults.map((q, i)=>(
                        <tr key={i}>
                          <td>{q.phase_name}</td>
                          <td>{q.weight_percent}</td>
                          <td>{q.lattice_params}</td>
                          <td>{q.crystallite_size_nm}</td>
                          <td>{q.confidence_score}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              </>
            )}

            {/* Show recommended vs final settings for debugging */}
            <Divider sx={{ my:2 }}/>
            <Typography variant="subtitle2">Recommended Settings (GPT)</Typography>
            <pre style={{ fontSize:'0.7rem' }}>
              {JSON.stringify(analysisResult.recommendedSettings, null, 2)}
            </pre>
            <Divider sx={{ my:2 }}/>
            <Typography variant="subtitle2">Final Settings (Merged)</Typography>
            <pre style={{ fontSize:'0.7rem' }}>
              {JSON.stringify(analysisResult.finalSettings, null, 2)}
            </pre>
          </Box>
        )}
      </Drawer>

      {/* MAIN CONTENT */}
      <MainContent leftOpen={leftDrawerOpen} rightOpen={rightDrawerOpen}>
        <Container maxWidth="lg" sx={{ py:2 }}>
          {errorMessage && (
            <Alert severity="error" sx={{ mb:2 }}>
              {errorMessage}
            </Alert>
          )}
          {isLoading && (
            <Box sx={{
              display:'flex',
              flexDirection:'column',
              alignItems:'center',
              justifyContent:'center',
              height:'calc(100vh - 64px)'
            }}>
              <CircularProgress/>
              <Typography variant="body2" mt={1}>Processing...</Typography>
            </Box>
          )}

          {/* If no data yet */}
          {!isLoading && !analysisResult && !clusterResult && !simulationResult && (
            <Card variant="outlined" sx={{ maxWidth:600, mx:'auto', mt:4 }}>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Welcome to the Enhanced XRD Dashboard
                </Typography>
                <Divider sx={{ my:2 }}/>
                <Typography variant="body1">
                  Now includes GPT-based numeric settings recommendations!  
                  If you leave background/smoothing settings blank, GPT will choose them
                  based on data length, intensity ranges, etc.
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* If cluster */}
          {clusterResult && !isLoading && (
            <Card variant="outlined" sx={{ mt:4, maxWidth:900, mx:'auto' }}>
              <CardContent>
                <Typography variant="h6">Cluster Results</Typography>
                <Divider sx={{ my:2 }}/>
                {clusterResult.clusters?.map((c, i)=>(
                  <Typography key={i} variant="body2" sx={{ mb:1 }}>
                    {c.filename} ={'>'} {c.cluster_label}: {c.explanation}
                  </Typography>
                ))}
                {clusterResult.finalReport && (
                  <>
                    <Divider sx={{ my:2 }}/>
                    <ReportDisplay text={clusterResult.finalReport}/>
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* If simulation */}
          {simulationResult && !isLoading && (
            <Card variant="outlined" sx={{ mt:4, maxWidth:900, mx:'auto' }}>
              <CardContent>
                <Typography variant="h6">Simulated Pattern</Typography>
                <Divider sx={{ my:2 }}/>
                {simulationResult.parsedData?.length>0 ? (
                  <Box sx={{ width:'100%', height:300 }}>
                    <ResponsiveContainer>
                      <LineChart data={simulationResult.parsedData}>
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="two_theta"/>
                        <YAxis dataKey="intensity"/>
                        <Tooltip/>
                        <Legend/>
                        <Line type="monotone" dataKey="intensity" stroke="#82ca9d" dot={false} name="Sim"/>
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <Typography variant="body2">No simulation data returned.</Typography>
                )}
                {simulationResult.finalReport && (
                  <>
                    <Divider sx={{ my:2 }}/>
                    <ReportDisplay text={simulationResult.finalReport}/>
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* Main Analysis */}
          {!isLoading && analysisResult && (
            <Box sx={{ width:'100%', background:'#fff', borderRadius:2, p:1, mt:2 }}>
              <ResponsiveGridLayout
                className="layout"
                layouts={layouts}
                breakpoints={{ lg:1200, md:996, sm:768, xs:480, xxs:0 }}
                cols={{ lg:12, md:12, sm:6, xs:4, xxs:2 }}
                rowHeight={30}
                margin={[10,10]}
                isResizable={false}
                isDraggable={false}
                compactType="vertical"
                style={{ background:'#fff', borderRadius:'4px' }}
              >
                <div key="diffPattern" style={{ background:'#fff', border:'1px solid #ddd', borderRadius:4, padding:10 }}>
                  <Typography variant="subtitle1" sx={{ mb:1 }}>
                    Multi-Step Diffraction Pattern
                  </Typography>
                  <Divider sx={{ mb:2 }}/>
                  {chartDataExists ? (
                    <Box sx={{ width:'100%', height:300 }}>
                      <ResponsiveContainer>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3"/>
                          <XAxis dataKey="two_theta" label={{ value:'2θ (deg)', position:'insideBottomRight' }}/>
                          <YAxis/>
                          <Tooltip/>
                          <Legend/>
                          <Line type="monotone" dataKey="Raw" stroke="#666666" dot={false} />
                          <Line type="monotone" dataKey="Calib" stroke="#ff9800" dot={false}/>
                          <Line type="monotone" dataKey="BG" stroke="#8bc34a" dot={false}/>
                          <Line type="monotone" dataKey="Smooth" stroke="#9c27b0" dot={false}/>
                          <Line type="monotone" dataKey="Kalpha" stroke="#00bcd4" dot={false}/>
                          <Line type="monotone" dataKey="Fitted" stroke="#f44336" dot={false}/>
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  ) : (
                    <Typography variant="body2">No chart data available.</Typography>
                  )}
                </div>

                <div key="finalReport" style={{ background:'#fff', border:'1px solid #ddd', borderRadius:4, padding:10 }}>
                  <Typography variant="subtitle1" sx={{ mb:1 }}>Final Report</Typography>
                  <Divider sx={{ mb:2 }}/>
                  {analysisResult.finalReport ? (
                    <ReportDisplay text={analysisResult.finalReport}/>
                  ) : (
                    <Typography variant="body2">No final report from GPT.</Typography>
                  )}
                </div>
              </ResponsiveGridLayout>
            </Box>
          )}
        </Container>
      </MainContent>
    </RootBox>
  );
}
