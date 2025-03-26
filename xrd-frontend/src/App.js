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
  Container
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
  Line,
  BarChart,
  Bar
} from 'recharts';

const ResponsiveGridLayout = WidthProvider(Responsive);

/** Splits multiline text into paragraphs. */
function ReportDisplay({ text }) {
  if (!text) return null;
  const paragraphs = text.split(/\n\s*\n/);
  return (
    <Box sx={{ mt: 1 }}>
      {paragraphs.map((p, idx) => (
        <Typography key={idx} variant="body2" paragraph sx={{ whiteSpace: 'pre-wrap' }}>
          {p}
        </Typography>
      ))}
    </Box>
  );
}

/**
 * Combine multiple data sets (parsed, bg, smoothed, stripped, fitted) into a single array
 * of objects for multi-line chart.
 */
function unifyDataMultiSeries({
  parsedData,
  bgCorrectedData,
  smoothedData,
  strippedData,
  fittedPeaks
}) {
  const map = new Map();

  function addSeries(dataArray, seriesKey) {
    dataArray.forEach(d => {
      const x = d.two_theta;
      if (!map.has(x)) {
        map.set(x, { two_theta: x });
      }
      map.get(x)[seriesKey] = d.intensity;
    });
  }

  // Add each data set with a distinct key
  addSeries(parsedData, 'Raw');
  addSeries(bgCorrectedData, 'BG');
  addSeries(smoothedData, 'Smooth');
  addSeries(strippedData, 'Kalpha');
  fittedPeaks?.forEach(fp => {
    const x = fp.two_theta;
    if (!map.has(x)) {
      map.set(x, { two_theta: x });
    }
    // We'll store the fitted intensities in "Fitted"
    map.get(x).Fitted = fp.intensity;
  });

  // Convert to array and sort by two_theta
  return Array.from(map.values()).sort((a, b) => a.two_theta - b.two_theta);
}

const LEFT_DRAWER_WIDTH = 260;
const RIGHT_DRAWER_WIDTH = 300;

/** Root container that flexes horizontally. */
const RootBox = styled('div')({
  display: 'flex',
  minHeight: '100vh',
  backgroundColor: '#f9f9f9'
});

/** Main content area that adjusts margin based on drawer open states. */
const MainContent = styled('main')(({ theme, leftOpen, rightOpen }) => ({
  flexGrow: 1,
  marginTop: theme.spacing(8),
  marginLeft: leftOpen ? LEFT_DRAWER_WIDTH : 0,
  marginRight: rightOpen ? RIGHT_DRAWER_WIDTH : 0,
  transition: theme.transitions.create(['margin'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen
  }),
}));

export default function App() {
  // Drawer states
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(true);
  const [rightDrawerOpen, setRightDrawerOpen] = useState(true);

  // Single file, multi-file, simulation
  const [singleFile, setSingleFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [multiFiles, setMultiFiles] = useState([]);
  const [clusterResult, setClusterResult] = useState(null);
  const [simulationText, setSimulationText] = useState('');
  const [simulationResult, setSimulationResult] = useState(null);

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Combine data for the multi-line chart
  const chartData = unifyDataMultiSeries({
    parsedData: analysisResult?.parsedData || [],
    bgCorrectedData: analysisResult?.bgCorrectedData || [],
    smoothedData: analysisResult?.smoothedData || [],
    strippedData: analysisResult?.strippedData || [],
    fittedPeaks: analysisResult?.fittedPeaks || []
  });

  // Build a bar chart array from quant results
  const quantData = analysisResult?.quantResults?.map(q => ({
    name: q.phase_name,
    wtPercent: q.weight_percent
  })) || [];

  // React Grid Layout configuration
  const layouts = {
    lg: [
      { i: 'multiLineChart', x: 0, y: 0, w: 8, h: 12 },
      { i: 'phaseComp', x: 8, y: 0, w: 4, h: 12 },
      { i: 'finalReport', x: 0, y: 12, w: 12, h: 8 }
    ],
    md: [
      { i: 'multiLineChart', x: 0, y: 0, w: 8, h: 12 },
      { i: 'phaseComp', x: 0, y: 12, w: 6, h: 12 },
      { i: 'finalReport', x: 6, y: 12, w: 6, h: 8 }
    ],
    sm: [
      { i: 'multiLineChart', x: 0, y: 0, w: 6, h: 12 },
      { i: 'phaseComp', x: 0, y: 12, w: 6, h: 12 },
      { i: 'finalReport', x: 0, y: 24, w: 6, h: 8 }
    ]
  };

  // Key endpoint
  const API_BASE = 'https://xrd-backend-enuq.onrender.com';
  // Drawer toggles
  const toggleLeftDrawer = () => setLeftDrawerOpen(!leftDrawerOpen);
  const toggleRightDrawer = () => setRightDrawerOpen(!rightDrawerOpen);

  /*******************************************************
   * Single-File
   *******************************************************/
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
    setClusterResult(null);
    setSimulationResult(null);
    setErrorMessage('');

    try {
      const formData = new FormData();
      formData.append('xrdFile', singleFile);
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

  /*******************************************************
   * Multi-file Clustering
   *******************************************************/
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
    } catch (err) {
      console.error(err);
      setErrorMessage('Cluster failed: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /*******************************************************
   * Simulation
   *******************************************************/
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
      if (!resp.ok) throw new Error(`Simulation error: ${resp.statusText}`);
      const data = await resp.json();
      setSimulationResult(data);
    } catch (err) {
      console.error(err);
      setErrorMessage('Simulation error: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  /*******************************************************
   * RENDER
   *******************************************************/
  return (
    <RootBox>
      <AppBar position="fixed" sx={{ zIndex: theme => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={toggleLeftDrawer} sx={{ mr: 2 }}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            XRD Dashboard (Numeric + GPT)
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
            width: LEFT_DRAWER_WIDTH,
            boxSizing: 'border-box',
            backgroundColor: '#f5f5f5',
            p: 2,
            pt: 8
          }
        }}
      >
        <Typography variant="h6" gutterBottom color="primary">
          XRD Tools
        </Typography>
        <Divider sx={{ mb: 2 }} />

        <Typography variant="subtitle1">One-Click Analysis</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon />} sx={{ my: 1 }}>
          Single File
          <input hidden type="file" accept=".xy,.txt" onChange={handleSingleFileChange} />
        </Button>
        {singleFile && <Typography variant="body2">{singleFile.name}</Typography>}
        <Button variant="contained" onClick={runAnalyze} disabled={!singleFile || isLoading} fullWidth>
          Analyze
        </Button>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle1">Multi-File Cluster</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon />} sx={{ my: 1 }}>
          Select Files
          <input hidden type="file" multiple accept=".xy,.txt" onChange={handleMultiFileChange} />
        </Button>
        {multiFiles.length > 0 && (
          <Typography variant="body2">{multiFiles.map(f => f.name).join(', ')}</Typography>
        )}
        <Button variant="contained" onClick={runCluster} disabled={!multiFiles.length || isLoading} fullWidth>
          Cluster
        </Button>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle1">Simulation</Typography>
        <TextField
          multiline
          rows={4}
          variant="outlined"
          placeholder="Enter structure..."
          value={simulationText}
          onChange={e => setSimulationText(e.target.value)}
          sx={{ width: '100%', mb: 1 }}
        />
        <Button variant="contained" onClick={runSimulate} disabled={!simulationText.trim() || isLoading} fullWidth>
          Simulate
        </Button>
      </Drawer>

      {/* RIGHT DRAWER - PERSISTENT */}
      <Drawer
        variant="persistent"
        anchor="right"
        open={rightDrawerOpen}
        sx={{
          '& .MuiDrawer-paper': {
            width: RIGHT_DRAWER_WIDTH,
            boxSizing: 'border-box',
            backgroundColor: '#f5f5f5',
            p: 2,
            pt: 8
          }
        }}
      >
        <Typography variant="h6" sx={{ mb: 2 }} color="primary">
          Detailed Results
        </Typography>
        {!analysisResult ? (
          <Typography variant="body2" color="text.secondary">
            No analysis data yet.
          </Typography>
        ) : (
          <Box sx={{ overflowY: 'auto', maxHeight: '80vh' }}>
            {/* Show phases */}
            {analysisResult.phases?.length > 0 && (
              <>
                <Typography variant="subtitle1">Phases Identified</Typography>
                {analysisResult.phases.map((ph, i) => (
                  <Typography key={i} variant="body2">
                    {ph.phase_name} (confidence: {(ph.confidence*100).toFixed(1)}%)
                  </Typography>
                ))}
                <Divider sx={{ my: 2 }} />
              </>
            )}

            {/* Fitted peaks */}
            {analysisResult.fittedPeaks?.length > 0 && (
              <>
                <Typography variant="subtitle1">Fitted Peaks</Typography>
                <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                  <table style={{ width: '100%', fontSize: '0.8rem' }}>
                    <thead>
                      <tr>
                        <th>2θ</th>
                        <th>Intensity</th>
                        <th>FWHM</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisResult.fittedPeaks.map((fp, idx) => (
                        <tr key={idx}>
                          <td>{fp.two_theta}</td>
                          <td>{fp.intensity}</td>
                          <td>{fp.fwhm}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
                <Divider sx={{ my: 2 }} />
              </>
            )}

            {/* Quant */}
            {analysisResult.quantResults?.length > 0 && (
              <>
                <Typography variant="subtitle1">Quantitative Analysis</Typography>
                <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                  <table style={{ width: '100%', fontSize: '0.8rem' }}>
                    <thead>
                      <tr>
                        <th>Phase</th>
                        <th>wt%</th>
                        <th>Lattice</th>
                        <th>Crystallite (nm)</th>
                        <th>Conf.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisResult.quantResults.map((qr, i) => (
                        <tr key={i}>
                          <td>{qr.phase_name}</td>
                          <td>{qr.weight_percent}</td>
                          <td>{qr.lattice_params}</td>
                          <td>{qr.crystallite_size_nm}</td>
                          <td>{qr.confidence_score}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              </>
            )}
          </Box>
        )}
      </Drawer>

      {/* MAIN CONTENT */}
      <MainContent leftOpen={leftDrawerOpen} rightOpen={rightDrawerOpen}>
        <Container maxWidth="lg" sx={{ py: 2 }}>
          {errorMessage && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {errorMessage}
            </Alert>
          )}

          {isLoading && (
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: 'calc(100vh - 64px)'
            }}>
              <CircularProgress />
              <Typography variant="body2" mt={1}>Processing...</Typography>
            </Box>
          )}

          {/* Welcome Card */}
          {!isLoading && !analysisResult && !clusterResult && !simulationResult && (
            <Card variant="outlined" sx={{ maxWidth: 600, mx: 'auto', mt: 4 }}>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Welcome to the XRD Dashboard
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="body1">
                  This demo uses numeric libraries for background subtraction, smoothing,
                  and Kα2 stripping, plus GPT for advanced analysis (peak detection,
                  phase ID, quantification, and final reporting).
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Show cluster */}
          {!isLoading && clusterResult && (
            <Card variant="outlined" sx={{ mt: 4, maxWidth: 900, mx: 'auto' }}>
              <CardContent>
                <Typography variant="h6">Cluster Results</Typography>
                <Divider sx={{ my: 2 }} />
                {clusterResult.clusters?.map((c, idx) => (
                  <Typography key={idx} variant="body2" sx={{ mb: 1 }}>
                    {c.filename} ={'>'} {c.cluster_label} : {c.explanation}
                  </Typography>
                ))}
                {clusterResult.finalReport && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <ReportDisplay text={clusterResult.finalReport} />
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* Show simulation */}
          {!isLoading && simulationResult && (
            <Card variant="outlined" sx={{ mt: 4, maxWidth: 900, mx: 'auto' }}>
              <CardContent>
                <Typography variant="h6">Simulated Pattern</Typography>
                <Divider sx={{ my: 2 }} />
                {simulationResult.parsedData?.length > 0 ? (
                  <Box sx={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                      <LineChart data={simulationResult.parsedData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="two_theta" />
                        <YAxis dataKey="intensity" />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="intensity" stroke="#82ca9d" name="Sim" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <Typography variant="body2">No simulation data returned.</Typography>
                )}
                {simulationResult.finalReport && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <ReportDisplay text={simulationResult.finalReport} />
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* Show the main analysis result */}
          {!isLoading && analysisResult && (
            <Box sx={{ width: '100%', background: '#fff', borderRadius: 2, p: 1, mt: 2 }}>
              <ResponsiveGridLayout
                className="layout"
                layouts={layouts}
                breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                cols={{ lg: 12, md: 12, sm: 6, xs: 4, xxs: 2 }}
                rowHeight={30}
                margin={[10, 10]}
                // Prevent resizing/drags so layout is fixed
                isResizable={false}
                isDraggable={false}
                compactType="vertical"
                style={{ background: '#fff', borderRadius: '4px' }}
              >
                {/* Multi-line chart */}
                <div key="multiLineChart" style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>
                    Diffraction Pattern (Numeric + GPT)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {chartData.length > 0 ? (
                    <Box sx={{ width: '100%', height: 300 }}>
                      <ResponsiveContainer>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="two_theta" label={{ value: '2θ (deg)', position: 'insideBottomRight', offset: 0 }} />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="Raw" stroke="#8884d8" dot={false} />
                          <Line type="monotone" dataKey="BG" stroke="#82ca9d" dot={false} />
                          <Line type="monotone" dataKey="Smooth" stroke="#ff7300" dot={false} />
                          <Line type="monotone" dataKey="Kalpha" stroke="#9933ff" dot={false} />
                          <Line type="monotone" dataKey="Fitted" stroke="#00b3b3" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  ) : (
                    <Typography variant="body2">No chart data.</Typography>
                  )}
                </div>

                {/* Phase Composition Panel (Bar chart of wt%) */}
                <div key="phaseComp" style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Phase Composition (wt%)</Typography>
                  <Divider sx={{ mb: 2 }} />
                  {quantData.length > 0 ? (
                    <Box sx={{ width: '100%', height: 300 }}>
                      <ResponsiveContainer>
                        <BarChart data={quantData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="wtPercent" fill="#8884d8" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  ) : (
                    <Typography variant="body2">
                      No quantitative data from GPT.
                    </Typography>
                  )}
                </div>

                {/* Final Report Panel */}
                <div key="finalReport" style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Final Report</Typography>
                  <Divider sx={{ mb: 2 }} />
                  {analysisResult.finalReport ? (
                    <ReportDisplay text={analysisResult.finalReport} />
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
