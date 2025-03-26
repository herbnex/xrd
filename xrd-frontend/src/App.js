import React, { useState, useEffect, useRef } from 'react';
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
  Container,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell
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

/** Multi-step illusions for user while loading. */
const STAGES = [
  'Parsing XRD File with GPT...',
  'Background Subtraction (NumPy)...',
  'Smoothing (NumPy)...',
  'Kα2 Stripping (NumPy)...',
  'Peak Detection (GPT)...',
  'Pattern Decomposition (GPT)...',
  'Phase Identification (GPT)...',
  'Quantitative Analysis (GPT)...',
  'Error Detection (GPT)...',
  'Final Report (GPT)...'
];

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

/** Simple function to unify experimental vs. fitted data for charting. */
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

/** Main content that auto-adjusts margin based on drawer states. */
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

  // File states
  const [singleFile, setSingleFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Progress states
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const progressTimerRef = useRef(null);

  // Error
  const [errorMessage, setErrorMessage] = useState('');

  // Stage-based illusions: run a timer that increments currentStage every 1 second
  useEffect(() => {
    if (isLoading) {
      setCurrentStage(0);
      progressTimerRef.current = setInterval(() => {
        setCurrentStage(prev => {
          if (prev < STAGES.length - 1) return prev + 1;
          return prev; // once we reach last stage, stay there
        });
      }, 1000);
    } else {
      // stop the interval
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
        progressTimerRef.current = null;
      }
      setCurrentStage(0);
    }
    return () => {
      // cleanup if user unmounts
      if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    };
  }, [isLoading]);

  // For chart
  const expData = analysisResult?.parsedData || [];
  const fittedPeaks = analysisResult?.fittedPeaks || [];
  const chartData = unifyExpFittedData(expData, fittedPeaks);

  // Layout config
  const layouts = {
    lg: [
      { i: 'stageProgress', x: 0, y: 0, w: 12, h: 2 },
      { i: 'diffPattern', x: 0, y: 2, w: 6, h: 10 },
      { i: 'parsedData', x: 6, y: 2, w: 6, h: 10 },
      { i: 'finalReport', x: 0, y: 12, w: 12, h: 6 }
    ]
  };

  // In a real app, change to your actual backend URL
  const API_BASE = 'http://localhost:8080';

  const toggleLeftDrawer = () => setLeftDrawerOpen(!leftDrawerOpen);
  const toggleRightDrawer = () => setRightDrawerOpen(!rightDrawerOpen);

  /*******************************************************
   * Single File Analysis
   *******************************************************/
  const handleSingleFileChange = (e) => {
    setSingleFile(e.target.files[0]);
    setAnalysisResult(null);
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

      const resp = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) throw new Error(`Analyze error: ${resp.statusText}`);
      const data = await resp.json();
      setAnalysisResult(data);
    } catch (err) {
      console.error(err);
      setErrorMessage('Analysis failed. ' + err.message);
    } finally {
      // once we have a response (success or error), stop loading
      setIsLoading(false);
    }
  };

  /*******************************************************
   * Render
   *******************************************************/
  return (
    <RootBox>
      <AppBar position="fixed" sx={{ zIndex: (th) => th.zIndex.drawer + 1 }}>
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

        <Typography variant="subtitle1" sx={{ mb: 1 }}>Single File</Typography>
        <Button variant="contained" component="label" startIcon={<UploadIcon />} sx={{ mb: 1 }}>
          Choose File
          <input hidden type="file" accept=".xy,.txt" onChange={handleSingleFileChange} />
        </Button>
        {singleFile && (
          <Typography sx={{ fontSize: 12, mb: 1 }}>{singleFile.name}</Typography>
        )}
        <Button variant="contained" onClick={runAnalyze} disabled={!singleFile || isLoading} fullWidth>
          Analyze
        </Button>
      </Drawer>

      {/* RIGHT DRAWER */}
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
        {analysisResult ? (
          <Box sx={{ overflowY: 'auto', maxHeight: '80vh' }}>
            {/* Show phases */}
            {analysisResult.phases?.length > 0 && (
              <>
                <Typography variant="subtitle1">Phases Identified</Typography>
                {analysisResult.phases.map((ph, idx) => (
                  <Typography key={idx} variant="body2">
                    {ph.phase_name} (confidence: {(ph.confidence * 100).toFixed(1)}%)
                  </Typography>
                ))}
                <Divider sx={{ my: 2 }} />
              </>
            )}
            {/* Show fitted peaks */}
            {analysisResult.fittedPeaks?.length > 0 && (
              <>
                <Typography variant="subtitle1">Fitted Peaks</Typography>
                <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>2θ</TableCell>
                        <TableCell>Intensity</TableCell>
                        <TableCell>FWHM</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {analysisResult.fittedPeaks.map((fp, i) => (
                        <TableRow key={i}>
                          <TableCell>{fp.two_theta}</TableCell>
                          <TableCell>{fp.intensity}</TableCell>
                          <TableCell>{fp.fwhm}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Box>
                <Divider sx={{ my: 2 }} />
              </>
            )}
            {/* Show quant results */}
            {analysisResult.quantResults?.length > 0 && (
              <>
                <Typography variant="subtitle1">Quantitative Analysis</Typography>
                <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Phase</TableCell>
                        <TableCell>wt%</TableCell>
                        <TableCell>Lattice</TableCell>
                        <TableCell>Size (nm)</TableCell>
                        <TableCell>Conf.</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {analysisResult.quantResults.map((q, i) => (
                        <TableRow key={i}>
                          <TableCell>{q.phase_name}</TableCell>
                          <TableCell>{q.weight_percent}</TableCell>
                          <TableCell>{q.lattice_params}</TableCell>
                          <TableCell>{q.crystallite_size_nm}</TableCell>
                          <TableCell>{q.confidence_score}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Box>
              </>
            )}
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No analysis data yet.
          </Typography>
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

          {/* Show stage-based progress if loading */}
          {isLoading && (
            <Box sx={{ textAlign: 'center', mb: 2 }}>
              <CircularProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                {STAGES[currentStage]}
              </Typography>
            </Box>
          )}

          {/* If no analysis yet */}
          {!isLoading && !analysisResult && (
            <Card variant="outlined" sx={{ mx: 'auto', mt: 4, maxWidth: 600 }}>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Welcome to the XRD Dashboard
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="body1">
                  Upload a single XRD file to analyze. You will see step-by-step progress in the center,
                  and final results (Phases, Fitted Peaks, etc.) in the right drawer.
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Once we have a result, display it */}
          {analysisResult && !isLoading && (
            <Box sx={{ width: '100%', background: '#fff', borderRadius: 2, p: 1 }}>
              <ResponsiveGridLayout
                className="layout"
                layouts={layouts}
                breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                cols={{ lg: 12, md: 12, sm: 6, xs: 4, xxs: 2 }}
                rowHeight={30}
                margin={[10, 10]}
                useCSSTransforms
                compactType="none"
              >
                {/* stageProgress is optional, but we can show final stage or a summary */}
                <div key="stageProgress" style={{ background: '#fafafa', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1">Analysis Steps Completed</Typography>
                  <Divider sx={{ my: 1 }} />
                  {STAGES.map((s, idx) => (
                    <Typography
                      key={idx}
                      variant="body2"
                      sx={{ color: idx <= currentStage ? 'text.primary' : 'text.disabled' }}
                    >
                      {s}
                    </Typography>
                  ))}
                </div>

                {/* Chart of Exp vs. Fitted */}
                <div key="diffPattern" style={{ background: '#fafafa', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Diffraction Pattern</Typography>
                  <Divider sx={{ mb: 2 }} />
                  {chartData.length > 0 ? (
                    <Box sx={{ width: '100%', height: 300 }}>
                      <ResponsiveContainer>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="two_theta" label={{ value: '2θ', position: 'insideBottomRight', offset: 0 }} />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="intensity" stroke="#8884d8" name="Experimental" dot={false} />
                          <Line type="monotone" dataKey="fittedIntensity" stroke="#82ca9d" name="Fitted" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  ) : (
                    <Typography variant="body2">No chart data found.</Typography>
                  )}
                </div>

                {/* Table of parsed data or other intermediate steps */}
                <div key="parsedData" style={{ background: '#fafafa', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Intermediate Steps</Typography>
                  <Divider sx={{ mb: 1 }} />
                  {/* Show some data about each stage if desired */}
                  <Typography variant="body2">
                    <strong>Parsed Points:</strong> {analysisResult.parsedData?.length || 0}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Background-Corrected Points:</strong> {analysisResult.bgCorrectedData?.length || 0}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Smoothed Points:</strong> {analysisResult.smoothedData?.length || 0}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Kα2-Stripped Points:</strong> {analysisResult.strippedData?.length || 0}
                  </Typography>
                </div>

                {/* Final Report */}
                <div key="finalReport" style={{ background: '#fafafa', border: '1px solid #ddd', borderRadius: 4, padding: 10 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Final Report</Typography>
                  <Divider sx={{ mb: 2 }} />
                  <ReportDisplay text={analysisResult.finalReport} />
                </div>
              </ResponsiveGridLayout>
            </Box>
          )}
        </Container>
      </MainContent>
    </RootBox>
  );
}
