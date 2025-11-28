import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

try:
    import sionna
    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
    import tensorflow as tf
    SIONNA_AVAILABLE = True
    print("✓ Sionna loaded successfully")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✓ Using GPU: {gpus[0]}")
    else:
        print("⚠ No GPU found, using CPU (slower)")
        
except ImportError:
    SIONNA_AVAILABLE = False
    print("⚠ Sionna not available, using simplified propagation model")

CONFIG = {
    'data_file': 'Wireless Communications - Data Collection - Data-2.csv',
    'scenes_dir': 'scenes',
    'results_dir': 'results',
    'use_sionna_rt': True,
    'fallback_to_simple': True,
    'tx_power_dbm': 20.0,
    'frequency_ghz': 5.0,
}

BUILDING_NAME_MAP = {
    'Kiewit': 'Kiewit',
    'Kauffman': 'Kauffman',
    'Bessey': 'Bessey',
    'Love Library': 'Love Library',
    'Love Library South': 'Love Library',
    'Adele Coryell': 'Love Library',
    'Carolyn Pope Edwards': 'Carolyn Pope Edwards',
    'Avery': 'Avery',
    'Coliseum': 'Coliseum',
    'Union': 'Union',
    'Oldfather': 'Oldfather',
    'Selleck': 'Selleck',
    'Brace': 'Brace',
    'Burnett': 'Burnett',
    'Memorial Stadium': 'Memorial Stadium'
}

BUILDING_TO_SCENE = {
    'Kiewit': 'kiewit',
    'Kauffman': 'kauffman',
    'Bessey': 'bessey',
    'Love Library': 'love_library',
    'Carolyn Pope Edwards': 'carolyn_pope_edwards',
    'Avery': 'avery',
    'Coliseum': 'coliseum',
    'Union': 'union',
    'Oldfather': 'oldfather',
    'Selleck': 'selleck',
    'Brace': 'brace',
    'Burnett': 'burnett',
    'Memorial Stadium': 'memorial_stadium'
}

def load_measurement_data(filepath):
    
    print(f"\n{'='*70}")
    print("LOADING MEASUREMENT DATA")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"⚠ Data file not found: {filepath}")
        print("Creating sample data from your provided measurements...")
        
        data = {
            'Hall': [
                'Kiewit', 'Kiewit', 'Kiewit', 'Kiewit',
                'Love Library', 'Love Library', 'Love Library', 'Love Library',
                'Selleck', 'Selleck',
                'Brace', 'Brace'
            ],
            'Location': [
                'A310', 'A310', 'Lobby', 'Lobby',
                'Adele Coryell Schmoker Learning Commons', 'Schmoker Learning Commons',
                'South Lobby', 'Lobby',
                'Dining Hall', 'Dining Hall',
                '210', '210'
            ],
            'SSID': [
                'eduroam', 'NU-Guest', 'eduroam', 'NU-Guest',
                'eduroam', 'NU-Guest', 'eduroam', 'NU-Guest',
                'eduroam', 'NU-Guest',
                'eduroam', 'NU-Guest'
            ],
            'Frequency (MHz)': [
                5200, 5745, 5784, 5785,
                5260, 5260, 5745, 5745,
                5745, 5745,
                5765, 5765
            ],
            'RSSI (dBm)': [
                -63, -59, -51, -54,
                -59, -55, -50, -47,
                -61, -62,
                -53, -50
            ],
            'Channel': [
                40, 149, 157, 157,
                52, 52, 149, 149,
                149, 149,
                153, 153
            ],
            'Time': [
                '11.19.25 @ 16:15', '11.19.25 @ 16:31', '11.19.25 @ 17:09', '11.19.25 @ 17:01',
                '11.19.25 @ 18:19', '11.19.25 @ 18:14', '11.19.25 @ 18:27', '11.19.25 @ 18:34',
                '11.19.25 @ 17:46', '11.19.25 @ 18:01',
                '11.21.25 @ 12:25', '11.21.25 @ 12:30'
            ]
        }
        
        df = pd.DataFrame(data)
        
        df.to_csv(filepath, index=False)
        print(f"✓ Created sample data file: {filepath}")
        
    else:
        try:
            try:
                df = pd.read_csv(filepath, quotechar='"', escapechar='\\', on_bad_lines='skip')
            except TypeError:
                df = pd.read_csv(filepath, quotechar='"', on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f"⚠ Error reading CSV: {e}")
            df = pd.read_csv(filepath, quotechar='"', engine='python', error_bad_lines=False, warn_bad_lines=False)
        print(f"✓ Loaded data from: {filepath}")
    
    df.columns = df.columns.str.strip()
    
    df['Hall'] = df['Hall'].ffill()
    
    df['RSSI (dBm)'] = pd.to_numeric(df['RSSI (dBm)'], errors='coerce')
    df['Frequency (MHz)'] = pd.to_numeric(df['Frequency (MHz)'], errors='coerce')
    
    df = df.dropna(subset=['RSSI (dBm)', 'Hall'])
    
    df['Hall'] = df['Hall'].map(lambda x: BUILDING_NAME_MAP.get(x, x) if x in BUILDING_NAME_MAP else x)
    
    df = df[df['Hall'].notna()]
    
    print(f"✓ Loaded {len(df)} measurements from {df['Hall'].nunique()} buildings")
    print(f"  Buildings: {', '.join(df['Hall'].unique())}")
    
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
            self.scene = load_scene(self.scene_file)
            self.scene.frequency = self.frequency_hz
            print(f"✓ Loaded scene: {self.scene_file}")
            return True
        except Exception as e:
            print(f"✗ Error loading scene: {e}")
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
        print(f"✓ Added transmitter at {position}")
    
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
        
        print(f"✓ Added {len(positions)} receivers")
    
    def compute_coverage(self, max_depth=5, num_samples=1e6):
        print(f"Running ray tracing (max_depth={max_depth}, samples={num_samples:.0e})...")
        
        try:
            paths = self.scene.compute_paths(
                max_depth=max_depth,
                num_samples=int(num_samples),
                los=True,
                reflection=True,
                diffraction=True
            )
            
            a, tau = paths.cir()
            
            rssi_values = []
            num_rx = a.shape[1]
            
            for rx_idx in range(num_rx):
                path_gains = a[0, rx_idx, 0, 0, 0, :]
                channel_power = tf.reduce_sum(tf.abs(path_gains) ** 2)
                
                tx_power_dbm = 20.0
                path_loss_db = -10 * tf.math.log(channel_power) / tf.math.log(10.0)
                rssi = tx_power_dbm - path_loss_db.numpy()
                
                rssi_values.append(float(rssi))
            
            print(f"✓ Ray tracing complete")
            return np.array(rssi_values)
            
        except Exception as e:
            print(f"✗ Ray tracing failed: {e}")
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
        print(f"SIMULATING: {building_name}")
        print(f"{'='*70}")
        
        building_data = self.data[self.data['Hall'] == building_name].copy()
        
        if len(building_data) == 0:
            print(f"⚠ No measurements for {building_name}")
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
            
            if not os.path.exists(scene_file):
                scene_file = os.path.join(
                    self.config['scenes_dir'],
                    f"{building_name.lower().replace(' ', '_')}.xml"
                )
            
            if os.path.exists(scene_file):
                print(f"Attempting Sionna ray tracing with {scene_file}...")
                try:
                    tracer = SionnaRayTracer(scene_file, avg_frequency)
                    if tracer.load_scene():
                        tracer.setup_transmitter(ap_position)
                        tracer.setup_receivers(rx_positions)
                        simulated_rssi = tracer.compute_coverage(max_depth=3, num_samples=5e5)
                        
                        if simulated_rssi is not None:
                            method = "Sionna Ray Tracing"
                        
                except Exception as e:
                    print(f"✗ Sionna ray tracing failed: {e}")
            else:
                print(f"⚠ Scene file not found: {scene_file}")
        
        if simulated_rssi is None and self.config['fallback_to_simple']:
            print("Using simplified propagation model...")
            model = SimplePropagationModel(avg_frequency, tx_power_dbm=20)
            
            simulated_rssi = []
            for rx_pos in rx_positions:
                distance = np.linalg.norm(rx_pos - ap_position)
                num_walls = int(distance / 10)
                rssi = model.calculate_rssi(distance, num_walls)
                simulated_rssi.append(rssi)
            
            simulated_rssi = np.array(simulated_rssi)
            method = "Simplified Model"
        
        if simulated_rssi is None:
            print("✗ Simulation failed")
            return None
        
        print(f"✓ Simulation complete using {method}")
        
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
    
  
    
    scenes_dir = CONFIG['scenes_dir']
    if not os.path.exists(scenes_dir) or len(os.listdir(scenes_dir)) == 0:
        print(f"⚠ No scene files found in '{scenes_dir}/'")
    
    data = load_measurement_data(CONFIG['data_file'])
    
    runner = SimulationRunner(data, CONFIG)
    
    results = runner.run_all_buildings()
    
    runner.create_visualizations()
    
    runner.print_summary()
    
    print(f"\n{'-'*73}")
    print("SIMULATION COMPLETE!")


if __name__ == "__main__":
    main()