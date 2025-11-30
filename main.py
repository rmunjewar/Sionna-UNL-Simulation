import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

# Set LLVM path for Sionna RT on macOS
if sys.platform == 'darwin' and 'DRJIT_LIBLLVM_PATH' not in os.environ:
    llvm_paths = [
        '/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib',
        '/opt/homebrew/opt/llvm/lib/libLLVM.dylib',
        '/usr/local/opt/llvm@20/lib/libLLVM.dylib',
        '/usr/local/opt/llvm/lib/libLLVM.dylib'
    ]
    for llvm_path in llvm_paths:
        if os.path.exists(llvm_path):
            os.environ['DRJIT_LIBLLVM_PATH'] = llvm_path
            break

try:
    import sionna
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    import tensorflow as tf
    SIONNA_AVAILABLE = True
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Found {len(gpus)} GPU device(s)")
    else:
        print("⚠ No GPU devices found, will use CPU")
        
except ImportError as e:
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    SIONNA_AVAILABLE = False
except Exception as e:
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    SIONNA_AVAILABLE = False

CONFIG = {
    'data_file': 'Wireless Communications - Data Collection - Data-2.csv',
    'scenes_dir': 'scenes',
    'results_dir': 'results',
    'use_sionna_rt': True,
    'fallback_to_simple': True,  # Set to False to force Sionna-only mode (will fail if Sionna unavailable)
    'tx_power_dbm': 20.0,
    'frequency_ghz': 5.0,
}

BUILDING_NAME_MAP = {
    'Kiewit': 'Kiewit',
    'Kauffman': 'Kauffman',
    'Adele Coryell': 'Adele Coryell',
    'Love Library South': 'Love Library South',
    'Selleck': 'Selleck',
    'Brace': 'Brace'
}

BUILDING_TO_SCENE = {
    'Kiewit': 'kiewit',
    'Kauffman': 'kauffman',
    'Adele Coryell': 'adele_coryell',
    'Love Library South': 'love_library_south',
    'Selleck': 'selleck',
    'Brace': 'brace'
}

def load_measurement_data(filepath):
    
    try:
        try:
            df = pd.read_csv(filepath, quotechar='"', escapechar='\\', on_bad_lines='skip')
        except TypeError:
            df = pd.read_csv(filepath, quotechar='"', on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        df = pd.read_csv(filepath, quotechar='"', engine='python', error_bad_lines=False, warn_bad_lines=False)
    print(f"Loaded data from: {filepath}")
    
    df.columns = df.columns.str.strip()
    
    df['Hall'] = df['Hall'].ffill()
    
    df['RSSI (dBm)'] = pd.to_numeric(df['RSSI (dBm)'], errors='coerce')
    df['Frequency (MHz)'] = pd.to_numeric(df['Frequency (MHz)'], errors='coerce')
    
    df = df.dropna(subset=['RSSI (dBm)', 'Hall'])
    
    df['Hall'] = df['Hall'].map(lambda x: BUILDING_NAME_MAP.get(x, x) if x in BUILDING_NAME_MAP else x)
    
    df = df[df['Hall'].notna()]
    
    print(f" Loaded {len(df)} measurements from {df['Hall'].nunique()} buildings")
    print(f"Buildings: {', '.join(df['Hall'].unique())}")
    
    return df

class SimplePropagationModel:
    
    def __init__(self, frequency_mhz=5200, tx_power_dbm=20):
        self.frequency_hz = frequency_mhz * 1e6
        self.tx_power_dbm = tx_power_dbm
        self.c = 3e8
        
    def calculate_rssi(self, distance, num_walls=0):
        
        if distance < 1.0:
            distance = 1.0
        
        fspl_1m = 20 * np.log10(self.frequency_hz) + 20 * np.log10(1.0) + \
                  20 * np.log10(4 * np.pi / self.c)
        
        path_loss_exponent = 3.5
        path_loss = fspl_1m + 10 * path_loss_exponent * np.log10(distance)
        
        wall_loss_db = 6.0
        total_loss = path_loss + (num_walls * wall_loss_db)
        
        rssi = self.tx_power_dbm - total_loss
        
        shadow_std = 4.0
        rssi += np.random.normal(0, shadow_std)
        
        return rssi

class SionnaRayTracer:
    
    def __init__(self, scene_file, frequency_mhz=5200):
        self.scene_file = scene_file
        self.frequency_hz = frequency_mhz * 1e6
        self.scene = None
        
    def load_scene(self):
        try:
            # Ensure we have an absolute path
            if self.scene_file is None:
                raise ValueError("Scene file path is None")
            
            scene_path = os.path.abspath(self.scene_file)
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene file does not exist: {scene_path}")
            
            self.scene = load_scene(scene_path)
            self.scene.frequency = self.frequency_hz
            print(f"Loaded scene: {scene_path}")
            return True
        except Exception as e:
            print(f"Error loading scene: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_transmitter(self, position):
        self.scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )
        
        tx = Transmitter(
            name="wifi_ap",
            position=position,
            orientation=[0, 0, 0]
        )
        self.scene.add(tx)
        print(f"Added transmitter at {position}")
    
    def setup_receivers(self, positions):
        self.scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )
        
        for i, pos in enumerate(positions):
            rx = Receiver(
                name=f"rx_{i}",
                position=pos,
                orientation=[0, 0, 0]
            )
            self.scene.add(rx)
        
        print(f"Added {len(positions)} receivers")
    
    def compute_coverage(self, max_depth=5, num_samples=1e6):
        print(f"Running ray tracing (max_depth={max_depth}, samples={num_samples:.0e})...")
        
        try:
            # Try using PathSolver if compute_paths is not available on scene
            try:
                # First try the direct method if available
                paths = self.scene.compute_paths(
                    max_depth=max_depth,
                    num_samples=int(num_samples),
                    los=True,
                    reflection=True,
                    diffraction=True
                )
            except AttributeError:
                # Fall back to PathSolver
                solver = PathSolver()
                paths = solver(self.scene)
            
            a, tau = paths.cir(out_type="numpy")
            
            # a is a tuple of (real, imag) components
            if isinstance(a, (tuple, list)) and len(a) == 2:
                a_real, a_imag = a
                # Convert to complex array
                a_complex = a_real + 1j * a_imag
            else:
                a_complex = a
            
            rssi_values = []
            # Handle different possible shapes: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
            if len(a_complex.shape) == 6:
                num_rx = a_complex.shape[0]
                for rx_idx in range(num_rx):
                    # Extract path gains for this receiver (assuming single TX, single antenna)
                    path_gains = a_complex[rx_idx, 0, 0, 0, :, 0]  # [num_paths]
                    channel_power = np.sum(np.abs(path_gains) ** 2)
                    
                    tx_power_dbm = 20.0
                    if channel_power > 0:
                        path_loss_db = -10 * np.log10(channel_power)
                        rssi = tx_power_dbm - path_loss_db
                    else:
                        rssi = -100.0  # Very weak signal
                    
                    rssi_values.append(float(rssi))
            else:
                # Fallback: try to extract from whatever shape we have
                print(f"Warning: Unexpected CIR shape {a_complex.shape}, attempting to extract RSSI...")
                # Sum over all dimensions except the path dimension
                channel_power = np.sum(np.abs(a_complex) ** 2)
                if channel_power > 0:
                    path_loss_db = -10 * np.log10(channel_power)
                    rssi = 20.0 - path_loss_db
                    rssi_values = [float(rssi)] * len(self.scene.receivers)
                else:
                    rssi_values = [-100.0] * len(self.scene.receivers)
            
            print(f"Ray tracing complete")
            return np.array(rssi_values)
            
        except Exception as e:
            print(f"Ray tracing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

class SimulationRunner:
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.results = {}
        
        os.makedirs(config['results_dir'], exist_ok=True)
    
    def weighted_centroid_localization(self, rx_positions, rssi_values):
        weights = 10 ** (rssi_values / 10.0)
        weights = weights / np.sum(weights)
        weighted_positions = rx_positions * weights[:, np.newaxis]
        estimated_pos = np.sum(weighted_positions, axis=0)
        estimated_pos[2] = 3.5
        return estimated_pos
    
    def estimate_ap_location(self, building_data, rx_positions=None, rssi_values=None):
        if rx_positions is None:
            rx_positions = self.estimate_rx_positions(building_data)
        if rssi_values is None:
            rssi_values = building_data['RSSI (dBm)'].values
        
        if len(rssi_values) > 0:
            return self.weighted_centroid_localization(rx_positions, rssi_values)
        else:
            return [0.0, 0.0, 3.0]
    
    def estimate_rx_positions(self, building_data):
        # TODO: Map 'Location' column from CSV to actual X/Y coordinates relative to the building floor plan.
        # Current circular logic creates inaccurate error metrics. Need to map location names (e.g., "A310", 
        # "Lobby", "Stairwell") to real-world coordinates within each building's geometry.
        locations = building_data['Location'].values
        positions = []
        
        for i, loc in enumerate(locations):
            angle = 2 * np.pi * i / len(locations)
            radius = 10 + (i % 3) * 5
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 1.5
            
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def simulate_building(self, building_name):
        
        print(f"\n{'='*70}")
        print(f"Simulating: {building_name}")
        print(f"{'='*70}")
        
        building_data = self.data[self.data['Hall'] == building_name].copy()
        
        if len(building_data) == 0:
            print(f"No measurements for {building_name}")
            return None
        
        print(f"Found {len(building_data)} measurements")
        
        measured_rssi = building_data['RSSI (dBm)'].values
        frequencies = building_data['Frequency (MHz)'].values
        avg_frequency = np.mean(frequencies)
        
        rx_positions = self.estimate_rx_positions(building_data)
        ap_position = self.estimate_ap_location(building_data, rx_positions, measured_rssi)
        
        print(f"AP position: {ap_position}")
        print(f"Average frequency: {avg_frequency:.0f} MHz")
        
        simulated_rssi = None
        method = "Unknown"
        
        if SIONNA_AVAILABLE and self.config['use_sionna_rt']:
            scene_name = BUILDING_TO_SCENE.get(building_name, building_name.lower().replace(' ', '_'))
            scene_file = os.path.join(
                self.config['scenes_dir'],
                f"{scene_name}.xml"
            )
            scene_file = os.path.abspath(scene_file)
            
            if not os.path.exists(scene_file):
                alt_scene_file = os.path.join(
                    self.config['scenes_dir'],
                    f"{building_name.lower().replace(' ', '_')}.xml"
                )
                alt_scene_file = os.path.abspath(alt_scene_file)
                if os.path.exists(alt_scene_file):
                    scene_file = alt_scene_file
                else:
                    print(f"⚠ Scene file not found: {scene_file}")
                    print(f"   Also checked: {alt_scene_file}")
            
            if os.path.exists(scene_file):
                try:
                    tracer = SionnaRayTracer(scene_file, avg_frequency)
                    if tracer.load_scene():
                        tracer.setup_transmitter(ap_position)
                        tracer.setup_receivers(rx_positions)
                        simulated_rssi = tracer.compute_coverage(max_depth=3, num_samples=5e5)
                        
                        if simulated_rssi is not None:
                            method = "Sionna Ray Tracing"
                        
                except Exception as e:
                    print(f"❌ Sionna ray tracing failed: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Scene file not found: {scene_file}")
        
        if simulated_rssi is None:
            if not SIONNA_AVAILABLE:
                print("⚠ Sionna not available - cannot use ray tracing")
                if self.config['fallback_to_simple']:
                    print("Using simplified propagation model (fallback enabled)")
                    model = SimplePropagationModel(avg_frequency, tx_power_dbm=20)
                    
                    simulated_rssi = []
                    for rx_pos in rx_positions:
                        distance = np.linalg.norm(rx_pos - ap_position)
                        num_walls = int(distance / 10)
                        rssi = model.calculate_rssi(distance, num_walls)
                        simulated_rssi.append(rssi)
                    
                    simulated_rssi = np.array(simulated_rssi)
                    method = "Simplified Model"
                else:
                    print("❌ Simulation failed: Sionna unavailable and fallback disabled")
                    print("   To enable fallback, set CONFIG['fallback_to_simple'] = True")
                    print("   Or install Sionna: pip install sionna")
                    return None
            elif not os.path.exists(scene_file):
                print(f"⚠ Scene file not found: {scene_file}")
                if self.config['fallback_to_simple']:
                    print("Using simplified propagation model (fallback enabled)")
                    model = SimplePropagationModel(avg_frequency, tx_power_dbm=20)
                    
                    simulated_rssi = []
                    for rx_pos in rx_positions:
                        distance = np.linalg.norm(rx_pos - ap_position)
                        num_walls = int(distance / 10)
                        rssi = model.calculate_rssi(distance, num_walls)
                        simulated_rssi.append(rssi)
                    
                    simulated_rssi = np.array(simulated_rssi)
                    method = "Simplified Model"
                else:
                    print("❌ Simulation failed: Scene file missing and fallback disabled")
                    return None
            else:
                print("❌ Simulation failed: Unknown error")
                return None
        
        print(f"Simulation complete using {method}")
        
        metrics = self.calculate_metrics(simulated_rssi, measured_rssi)
        
        result = {
            'building': building_name,
            'method': method,
            'measured_rssi': measured_rssi,
            'simulated_rssi': simulated_rssi,
            'locations': building_data['Location'].values,
            'metrics': metrics,
            'ap_position': ap_position,
            'rx_positions': rx_positions
        }
        
        self.results[building_name] = result
        
        self.print_metrics(building_name, metrics)
        
        return result
    
    def calculate_metrics(self, simulated, measured):
        error = simulated - measured
        
        metrics = {
            'MAE': np.mean(np.abs(error)),
            'RMSE': np.sqrt(np.mean(error**2)),
            'Mean_Error': np.mean(error),
            'Std_Error': np.std(error),
            'Max_Error': np.max(np.abs(error)),
            'Correlation': np.corrcoef(simulated, measured)[0, 1] if len(simulated) > 1 else 0
        }
        
        return metrics
    
    def print_metrics(self, building, metrics):
        print(f"\nResults for {building}:")
        print(f"  MAE:         {metrics['MAE']:6.2f} dB")
        print(f"  RMSE:        {metrics['RMSE']:6.2f} dB")
        print(f"  Mean Error:  {metrics['Mean_Error']:6.2f} dB")
        print(f"  Std Error:   {metrics['Std_Error']:6.2f} dB")
        print(f"  Max Error:   {metrics['Max_Error']:6.2f} dB")
        print(f"  Correlation: {metrics['Correlation']:6.3f}")
    
    def run_all_buildings(self):
        buildings = self.data['Hall'].unique()
        
        print(f"\n{'='*70}")
        print(f"RUNNING SIMULATIONS FOR {len(buildings)} BUILDINGS")
        print(f"{'='*70}")
        
        for building in buildings:
            self.simulate_building(building)
        
        return self.results
    
    def create_visualizations(self):
        print(f"\n{'='*70}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        for building, result in self.results.items():
            self._plot_building_comparison(result)
            self._plot_scene_layout(result)
        
        self._plot_summary()
    
    def _plot_scene_layout(self, result):
        building = result['building']
        ap_position = result['ap_position']
        rx_positions = result['rx_positions']
        measured_rssi = result['measured_rssi']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.scatter(ap_position[0], ap_position[1], s=500, marker='^', 
                  color='red', label='Access Point', zorder=5, edgecolors='black', linewidths=2)
        
        scatter = ax.scatter(rx_positions[:, 0], rx_positions[:, 1], 
                           c=measured_rssi, s=200, cmap='RdYlGn_r', 
                           edgecolors='black', linewidths=1.5, zorder=4,
                           label='Measurement Points')
        
        for i, (pos, rssi) in enumerate(zip(rx_positions, measured_rssi)):
            ax.annotate(f'{rssi:.0f} dBm', 
                       (pos[0], pos[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', alpha=0.7))
        
        for rx_pos in rx_positions:
            ax.plot([ap_position[0], rx_pos[0]], 
                   [ap_position[1], rx_pos[1]], 
                   'k--', alpha=0.3, linewidth=1, zorder=1)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Measured RSSI (dBm)', fontsize=12)
        
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title(f'{building} - Scene Layout with Measurement Points', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        filename = os.path.join(
            self.config['results_dir'],
            f"{building.replace(' ', '_')}_scene_layout.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved scene layout: {filename}")
        plt.close()
    
    def _plot_building_comparison(self, result):
        building = result['building']
        measured = result['measured_rssi']
        simulated = result['simulated_rssi']
        locations = result['locations']
        metrics = result['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{building} - {result["method"]}', fontsize=16, fontweight='bold')
        
        short_locations = [loc[:20] + '...' if len(loc) > 20 else loc for loc in locations]
        
        axes[0, 0].scatter(measured, simulated, alpha=0.6, s=100)
        lim_min = min(measured.min(), simulated.min()) - 5
        lim_max = max(measured.max(), simulated.max()) + 5
        axes[0, 0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
                        label='Perfect prediction', linewidth=2)
        axes[0, 0].set_xlabel('Measured RSSI (dBm)', fontsize=12)
        axes[0, 0].set_ylabel('Simulated RSSI (dBm)', fontsize=12)
        axes[0, 0].set_title('Simulated vs Measured')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        error = simulated - measured
        axes[0, 1].hist(error, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        axes[0, 1].axvline(np.mean(error), color='g', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(error):.1f} dB')
        axes[0, 1].set_xlabel('Prediction Error (dB)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        x_pos = np.arange(len(locations))
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, measured, width, label='Measured', 
                      alpha=0.7, color='steelblue')
        axes[1, 0].bar(x_pos + width/2, simulated, width, label='Simulated', 
                      alpha=0.7, color='coral')
        axes[1, 0].set_xlabel('Location', fontsize=12)
        axes[1, 0].set_ylabel('RSSI (dBm)', fontsize=12)
        axes[1, 0].set_title('RSSI by Location')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(short_locations, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        axes[1, 1].axis('off')
        metrics_text = f"""
        Error Metrics:
        
        MAE:          {metrics['MAE']:.2f} dB
        RMSE:         {metrics['RMSE']:.2f} dB
        Mean Error:   {metrics['Mean_Error']:.2f} dB
        Std Error:    {metrics['Std_Error']:.2f} dB
        Max Error:    {metrics['Max_Error']:.2f} dB
        Correlation:  {metrics['Correlation']:.3f}
        
        Method: {result['method']}
        Measurements: {len(measured)}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        filename = os.path.join(
            self.config['results_dir'],
            f"{building.replace(' ', '_')}_comparison.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def _plot_summary(self):
        if len(self.results) == 0:
            return
        
        buildings = list(self.results.keys())
        mae_values = [self.results[b]['metrics']['MAE'] for b in buildings]
        rmse_values = [self.results[b]['metrics']['RMSE'] for b in buildings]
        corr_values = [self.results[b]['metrics']['Correlation'] for b in buildings]
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Summary: Simulated vs Measured WiFi Coverage Across UNL Campus', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        x_pos = np.arange(len(buildings))
        short_buildings = [b[:15] + '...' if len(b) > 15 else b for b in buildings]
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(x_pos, mae_values, color='steelblue', alpha=0.7)
        ax1.set_ylabel('MAE (dB)', fontsize=12)
        ax1.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(short_buildings, rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(np.mean(mae_values), color='r', linestyle='--', 
                   label=f'Average: {np.mean(mae_values):.2f} dB', linewidth=2)
        ax1.legend(fontsize=10)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x_pos, rmse_values, color='coral', alpha=0.7)
        ax2.set_ylabel('RMSE (dB)', fontsize=12)
        ax2.set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(short_buildings, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(np.mean(rmse_values), color='r', linestyle='--',
                   label=f'Average: {np.mean(rmse_values):.2f} dB', linewidth=2)
        ax2.legend(fontsize=10)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(x_pos, corr_values, color='mediumseagreen', alpha=0.7)
        ax3.set_ylabel('Correlation', fontsize=12)
        ax3.set_title('Measured vs Simulated Correlation', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(short_buildings, rotation=45, ha='right', fontsize=9)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(np.mean(corr_values), color='r', linestyle='--',
                   label=f'Average: {np.mean(corr_values):.3f}', linewidth=2)
        ax3.legend(fontsize=10)
        
        ax4 = fig.add_subplot(gs[1, :])
        all_measured = []
        all_simulated = []
        colors = plt.cm.tab20(np.linspace(0, 1, len(buildings)))
        
        for i, building in enumerate(buildings):
            measured = self.results[building]['measured_rssi']
            simulated = self.results[building]['simulated_rssi']
            all_measured.extend(measured)
            all_simulated.extend(simulated)
            ax4.scatter(measured, simulated, alpha=0.6, s=80, 
                       label=building[:20], color=colors[i])
        
        lim_min = min(min(all_measured), min(all_simulated)) - 5
        lim_max = max(max(all_measured), max(all_simulated)) + 5
        ax4.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', 
                label='Perfect prediction', linewidth=2)
        ax4.set_xlabel('Measured RSSI (dBm)', fontsize=12)
        ax4.set_ylabel('Simulated RSSI (dBm)', fontsize=12)
        ax4.set_title('All Buildings: Simulated vs Measured RSSI', 
                     fontsize=14, fontweight='bold')
        ax4.legend(ncol=3, fontsize=8, loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, 0])
        all_errors = []
        for building in buildings:
            measured = self.results[building]['measured_rssi']
            simulated = self.results[building]['simulated_rssi']
            errors = simulated - measured
            all_errors.extend(errors)
        
        sorted_errors = np.sort(all_errors)
        p = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax5.plot(sorted_errors, p, linewidth=2, color='steelblue')
        ax5.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax5.axvline(np.mean(all_errors), color='g', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_errors):.1f} dB')
        ax5.set_xlabel('Prediction Error (dB)', fontsize=12)
        ax5.set_ylabel('Cumulative Probability', fontsize=12)
        ax5.set_title('CDF of Prediction Errors', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(all_errors, bins=20, edgecolor='black', alpha=0.7, 
                color='skyblue', density=True)
        ax6.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax6.axvline(np.mean(all_errors), color='g', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_errors):.1f} dB')
        ax6.set_xlabel('Prediction Error (dB)', fontsize=12)
        ax6.set_ylabel('Density', fontsize=12)
        ax6.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        overall_mae = np.mean(mae_values)
        overall_rmse = np.mean(rmse_values)
        overall_corr = np.mean(corr_values)
        overall_mean_error = np.mean(all_errors)
        overall_std_error = np.std(all_errors)
        
        stats_text = f"""
        Overall Statistics:
        
        Average MAE:        {overall_mae:.2f} ± {np.std(mae_values):.2f} dB
        Average RMSE:       {overall_rmse:.2f} ± {np.std(rmse_values):.2f} dB
        Average Correlation: {overall_corr:.3f} ± {np.std(corr_values):.3f}
        
        Error Statistics:
        Mean Error:         {overall_mean_error:.2f} dB
        Std Deviation:      {overall_std_error:.2f} dB
        Min Error:          {np.min(all_errors):.2f} dB
        Max Error:          {np.max(all_errors):.2f} dB
        
        Total Measurements: {len(all_errors)}
        Buildings Analyzed: {len(buildings)}
        """
        ax7.text(0.1, 0.5, stats_text, fontsize=11, 
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        filename = os.path.join(self.config['results_dir'], 'summary.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    def print_summary(self):
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        
        if len(self.results) == 0:
            print("No results to summarize")
            return
        
        all_mae = [r['metrics']['MAE'] for r in self.results.values()]
        all_rmse = [r['metrics']['RMSE'] for r in self.results.values()]
        all_corr = [r['metrics']['Correlation'] for r in self.results.values()]
        
        print(f"\nAcross {len(self.results)} buildings:")
        print(f"  Average MAE:         {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} dB")
        print(f"  Average RMSE:        {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f} dB")
        print(f"  Average Correlation: {np.mean(all_corr):.3f} ± {np.std(all_corr):.3f}")
        
        print(f"\nBest performing building: {min(self.results, key=lambda x: self.results[x]['metrics']['MAE'])}")
        print(f"Worst performing building: {max(self.results, key=lambda x: self.results[x]['metrics']['MAE'])}")
        
        print(f"\nResults saved in: {self.config['results_dir']}/")

def main():
    VALID_BUILDINGS = {'Kiewit', 'Kauffman', 'Adele Coryell', 'Love Library South', 'Selleck', 'Brace'}
    
    # Print Sionna status
    if not SIONNA_AVAILABLE:
        print("⚠ WARNING: Sionna is not installed or failed to import")
        print("   The simulation will use simplified propagation model")
        print("   To install Sionna: pip install sionna")
        print("   Note: Sionna requires TensorFlow and may have compatibility issues on macOS\n")
    else:
        print("✓ Sionna is available - ray tracing enabled\n")
    
    scenes_dir = CONFIG['scenes_dir']
    if not os.path.exists(scenes_dir) or len(os.listdir(scenes_dir)) == 0:
        print(f"⚠ No scene files found in '{scenes_dir}/'")
    
    data = load_measurement_data(CONFIG['data_file'])
    
    # Validate buildings in data
    invalid_buildings = set(data['Hall'].unique()) - VALID_BUILDINGS
    if invalid_buildings:
        print(f"⚠ Warning: CSV contains unexpected buildings that will be ignored: {invalid_buildings}")
    
    data = data[data['Hall'].isin(VALID_BUILDINGS)]
    print(f"✓ Processing {len(VALID_BUILDINGS)} validated buildings")
    
    runner = SimulationRunner(data, CONFIG)
    
    results = runner.run_all_buildings()
    
    runner.create_visualizations()
    
    runner.print_summary()
    
    print(f"\n{'-'*73}")
    print("SIMULATION COMPLETE!")


if __name__ == "__main__":
    main()