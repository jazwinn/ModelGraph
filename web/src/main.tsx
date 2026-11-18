import '@xyflow/react/dist/style.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { useStore } from './lib/store';
import './styles.css';

// Handy for debugging and end-to-end tests: window.modelgraph.getState().
(window as unknown as { modelgraph: typeof useStore }).modelgraph = useStore;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
