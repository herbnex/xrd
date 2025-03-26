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

/** Combine Exp vs. Fitted data for an overlaid chart. */
function unifyExpFittedData(expData, fittedPeaks) {
  const map = new Map();
  expData.forEach(d => {
    map.set(d.two_theta, {
      two_theta: d.two_theta,
      intensity: d.intensity,
      fittedIntensity: undefined
    });
  });
  fittedPeaks?.forEach(fp => {
    if (!map.has(fp.two_theta)) {
      map.set(fp.two_theta, {
        two_theta: fp.two_theta,
        intensity: undefined,
        fittedIntensity: fp.intensity
      });
    } else {
      const existing = map.get(fp.two_theta);
      existing.fittedIntensity = fp.intensity;
    }
  });
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

export default function App() {
  // Left drawer always "temporary"
  const [leftDrawerOpen, setLeftDrawerOpen] = useState(true);
  // Right drawer is persistent (can toggle)
  const [rightDrawerOpen, setRightDrawerOpen] = useState(true);

  // File states
  const [singleFile, setSingleFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [multiFiles, setMultiFiles] = useState([]);
  const [clusterResult, setClusterResult] = useState(null);
  const [simulationText, setSimulationText] = useState('');
  const [simulationResult, setSimulationResult] = useState(null);

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Merge data
  const expData = analysisResult?.parsedData || [];
  const fittedPeaks = analysisResult?.fittedPeaks || [];
  const chartData = unifyExpFittedData(expData, fittedPeaks);

  const quantData = analysisResult?.quantResults?.map(q => ({
    name: q.phase_name,
    wtPercent: q.weight_percent
  })) || [];

  // React Grid Layout configuration
  const layouts = {
    lg: [
      { i: 'diffPattern', x: 0, y: 0, w: 6, h: 10 },
      { i: 'phaseComp', x: 6, y: 0, w: 6, h: 10 },
      { i: 'finalReport', x: 0, y: 10, w: 12, h: 8 }
    ],
    md: [
      { i: 'diffPattern', x: 0, y: 0, w: 6, h: 10 },
      { i: 'phaseComp', x: 6, y: 0, w: 6, h: 10 },
      { i: 'finalReport', x: 0, y: 10, w: 12, h: 8 }
    ],
    sm: [
      { i: 'diffPattern', x: 0, y: 0, w: 6, h: 10 },
      { i: 'phaseComp', x: 0, y: 10, w: 6, h: 10 },
      { i: 'finalReport', x: 0, y: 20, w: 6, h: 8 }
    ]
  };

  // IMPORTANT: Point to your local Flask server
  const API_BASE = 'http://127.0.0.1:8080';

  // Drawer toggles
  const toggleLeftDrawer = () => setLeftDrawerOpen(!leftDrawerOpen);
  const toggleRightDrawer = () => setRightDrawerOpen(!rightDrawerOpen);

  /*******************************************************
   * Single File
   *******************************************************/
  const handleSingleFileChange = (e) => {
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
      console.log("Analysis result:", data);
      setAnalysisResult(data);
    } catch (err) {
      console.error(err);
      setErrorMessage('Analysis failed.');
    } finally {
      setIsLoading(false);
    }
  };

  /*******************************************************
   * Multi-file
   *******************************************************/
  const handleMultiFileChange = (e) => {
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
      setErrorMessage('Cluster failed.');
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
      setErrorMessage('Simulation error.');
    } finally {
      setIsLoading(false);
    }
  };

  /*******************************************************
   * RENDER
   *******************************************************/
  return (
    <RootBox>
      <AppBar position="fixed" sx={{ zIndex: (th) => th.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={toggleLeftDrawer} sx={{ mr: 2 }}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            XRD Dashboard (All GPT)
          </Typography>
          <Button color="inherit" onClick={toggleRightDrawer}>
            {rightDrawerOpen ? 'Hide Phases' : 'Show Phases'}
          </Button>
        </Toolbar>
      </AppBar>

      {/* LEFT DRAWER - TEMPORARY */}
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
            pt: 8 // push content below app bar
          }
        }}
      >
        <Typography variant="h6" gutterBottom color="primary">
          XRD Tools
        </Typography>
        <Divider sx={{ mb: 2 }} />

        <Typography variant="subtitle1" sx={{ mb: 1 }}>One-Click Analysis</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon />} sx={{ mb: 1 }}>
          Single File
          {/* NOTE: no "accept" attribute => can upload any extension */}
          <input hidden type="file" onChange={handleSingleFileChange} />
        </Button>
        {singleFile && (
          <Typography sx={{ fontSize: 12, mb: 1 }}>{singleFile.name}</Typography>
        )}
        <Button variant="contained" onClick={runAnalyze} disabled={!singleFile} fullWidth>
          Analyze
        </Button>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle1" sx={{ mb: 1 }}>Multi-File Cluster</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon />} sx={{ mb: 1 }}>
          Select Files
          {/* again, no accept => any extension */}
          <input hidden type="file" multiple onChange={handleMultiFileChange} />
        </Button>
        {multiFiles.length > 0 && (
          <Typography sx={{ fontSize: 12, mb: 1 }}>
            {multiFiles.map(f => f.name).join(', ')}
          </Typography>
        )}
        <Button variant="contained" onClick={runCluster} disabled={!multiFiles.length} fullWidth>
          Cluster
        </Button>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle1" sx={{ mb: 1 }}>Simulation</Typography>
        <TextField
          multiline
          rows={4}
          variant="outlined"
          placeholder="Enter structure..."
          value={simulationText}
          onChange={(e) => setSimulationText(e.target.value)}
          sx={{ width: '100%', mb: 1 }}
        />
        <Button variant="contained" onClick={runSimulate} disabled={!simulationText.trim()} fullWidth>
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
          Phases / Rietveld
        </Typography>
        {!analysisResult && (
          <Typography variant="body2" color="text.secondary">
            No analysis data yet.
          </Typography>
        )}
        {analysisResult && (
          <Box sx={{ overflowY: 'auto' }}>
            {analysisResult.phases?.length > 0 && (
              <>
                <Typography variant="subtitle1">Phases Identified</Typography>
                {analysisResult.phases.map((ph, i) => (
                  <Typography key={i} variant="body2">
                    {ph.phase_name}<br />
                    Confidence: {(ph.confidence * 100).toFixed(1)}%
                  </Typography>
                ))}
              </>
            )}

            {analysisResult.fittedPeaks?.length > 0 && (
              <>
                <Typography variant="subtitle1" sx={{ mt: 2 }}>Fitted Peaks</Typography>
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
              </>
            )}

            {analysisResult.quantResults?.length > 0 && (
              <>
                <Typography variant="subtitle1" sx={{ mt: 2 }}>Quantitative Analysis</Typography>
                <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                  <table style={{ width: '100%', fontSize: '0.8rem' }}>
                    <thead>
                      <tr>
                        <th>Phase</th>
                        <th>wt%</th>
                        <th>Lattice</th>
                        <th>Crystallite(nm)</th>
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

      {/* MAIN CONTENT (center) */}
      <Box sx={{
        flexGrow: 1,
        pt: 8, // offset for AppBar
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        overflow: 'auto'
      }}>
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
                  Please upload a single XRD file to analyze, or multiple files to cluster. 
                  You can also run a simulation by entering a structure in the left drawer.
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* React Grid Layout for Analysis Data */}
          {!isLoading && analysisResult && (
            <Box sx={{ width: '100%', background: '#fff', borderRadius: '4px', p: 1 }}>
              <ResponsiveGridLayout
                className="layout"
                layouts={layouts}
                breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                cols={{ lg: 12, md: 12, sm: 6, xs: 4, xxs: 2 }}
                rowHeight={30}
                margin={[10, 10]}
                useCSSTransforms
                compactType="none"
                style={{ background: '#fff', borderRadius: '4px' }}
              >
                {/* Diffraction Pattern Panel */}
                <div key="diffPattern" style={{ background: '#fff', border: '1px solid #ddd', borderRadius: '4px', padding: '10px' }}>
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Diffraction Pattern
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {chartData.length > 0 ? (
                    <Box sx={{ width: '100%', height: '100%' }}>
                      <ResponsiveContainer>
                        <LineChart data={chartData} margin={{ top: 20, right: 20, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="two_theta" label={{ value: '2θ (deg)', position: 'insideBottomRight', offset: 0 }} />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="intensity" stroke="#8884d8" name="Exp." />
                          <Line type="monotone" dataKey="fittedIntensity" stroke="#82ca9d" name="Fitted" />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  ) : (
                    <Typography variant="body2">No chart data found.</Typography>
                  )}
                </div>

                {/* Phase Composition Panel */}
                <div key="phaseComp" style={{ background: '#fff', border: '1px solid #ddd', borderRadius: '4px', padding: '10px' }}>
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Phase Composition (wt%)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {quantData.length > 0 ? (
                    <Box sx={{ width: '100%', height: '100%' }}>
                      <ResponsiveContainer>
                        <BarChart data={quantData} margin={{ top: 20, right: 20, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="wtPercent" fill="#8884d8" name="wt%" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  ) : (
                    <Typography variant="body2">No composition data from GPT.</Typography>
                  )}
                </div>

                {/* Final Report Panel */}
                <div key="finalReport" style={{ background: '#fff', border: '1px solid #ddd', borderRadius: '4px', padding: '10px' }}>
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Final Report
                  </Typography>
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

          {/* Cluster Results */}
          {clusterResult && !isLoading && (
            <Card variant="outlined" sx={{ mt: 4, maxWidth: 900, mx: 'auto' }}>
              <CardContent>
                <Typography variant="h6">Cluster Results</Typography>
                <Divider sx={{ my: 2 }} />
                {clusterResult.clusters?.map((c, idx) => (
                  <Typography key={idx} variant="body2" sx={{ mb: 1 }}>
                    {c.filename} =&gt; cluster={c.cluster_label} - {c.explanation}
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

          {/* Simulation Results */}
          {simulationResult && !isLoading && (
            <Card variant="outlined" sx={{ mt: 4, maxWidth: 900, mx: 'auto' }}>
              <CardContent>
                <Typography variant="h6">Simulated Pattern</Typography>
                <Divider sx={{ my: 2 }} />
                {simulationResult.parsedData?.length > 0 ? (
                  <Box sx={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                      <LineChart data={simulationResult.parsedData} margin={{ top: 20, right: 20, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="two_theta" />
                        <YAxis dataKey="intensity" />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="intensity" stroke="#82ca9d" name="Sim." />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <Typography variant="body2">No simulation data from GPT</Typography>
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
        </Container>
      </Box>
    </RootBox>
  );
}
