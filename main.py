import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# Import MEASUREMENT_LOCATIONS from scene_builder
try:
    from scene_builder import MEASUREMENT_LOCATIONS
except ImportError:
    # Fallback: define it here if import fails
    MEASUREMENT_LOCATIONS = {}

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
    'data_file': 'Wireless Communications - Data Collection - Data-4.csv',
    'scenes_dir': 'scenes',
    'results_dir': 'results',
    'use_sionna_rt': True,
    'fallback_to_simple': True,  
    # Transmit power in dBm
    # Note: Real WiFi APs transmit at 15-20 dBm, but we use lower "effective" power
    # to compensate for simplified building geometry (missing furniture, people, equipment)
    # and other unmodeled losses. Adjusted based on calibration with measured data.
    # Value of -5 dBm accounts for ~25 dB of unmodeled losses (furniture, people, multi-floor effects)
    'tx_power_dbm': -5.0,  # Effective TX power (calibrated from measurements)
    'frequency_ghz': 5.0,
    # Auto-calibration: Adjust each building's simulation to match mean measured RSSI
    # This compensates for building-specific geometry quality differences
    # WARNING: When enabled, this uses measured data for calibration, so errors are artificially low
    # For true validation, set to False and find optimal global TX power
    'auto_calibrate': False,
    'MANUAL_AP_POSITIONS': {
        'Hamilton': [18.5, 22.0, 3.0]
    }
}

BUILDING_NAME_MAP = {
    'Kiewit': 'Kiewit',
    'Kauffman': 'Kauffman',
    'Adele Coryell': 'Adele Coryell',
    'Love Library South': 'Love Library South',
    'Selleck': 'Selleck',
    'Brace': 'Brace',
    'Hamilton': 'Hamilton',
    'Bessey': 'Bessey',
    'Union': 'Union',
    'Oldfather': 'Oldfather',
    'Burnett': 'Burnett',
    'Memorial Stadium': 'Memorial Stadium'
}

BUILDING_TO_SCENE = {
    'Kiewit': 'kiewit',
    'Kauffman': 'kauffman',
    'Adele Coryell': 'adele_coryell',
    'Love Library South': 'love_library_south',
    'Selleck': 'selleck',
    'Brace': 'brace',
    'Hamilton': 'hamilton',
    'Bessey': 'bessey',
    'Union': 'union',
    'Oldfather': 'oldfather',
    'Burnett': 'burnett',
    'Memorial Stadium': 'memorial_stadium'
}

def parse_scene_xml(xml_file):
    """Parse scene XML file to extract building geometry and materials"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Material mapping
        materials = {}
        for bsdf in root.findall('.//bsdf'):
            mat_id = bsdf.get('id', '')
            mat_type_elem = bsdf.find('.//string[@name="type"]')
            if mat_type_elem is not None:
                mat_type = mat_type_elem.get('value', 'concrete')
                materials[mat_id] = mat_type
        
        # Extract shapes (walls, floors, ceilings, doors, windows, rooms)
        geometry = {
            'walls': [],
            'floors': [],
            'ceilings': [],
            'doors': [],
            'windows': [],
            'obstacles': []
        }
        
        for shape in root.findall('.//shape'):
            shape_type = shape.get('type', '')
            shape_name = shape.get('name', '')
            shape_id = shape.get('id', '')
            
            # Get material reference
            mat_ref = shape.find('.//ref[@name="bsdf"]')
            mat_id = mat_ref.get('id', '') if mat_ref is not None else ''
            material = materials.get(mat_id, 'concrete')
            
            # Parse transform to get position and size
            transform = shape.find('.//transform[@name="to_world"]')
            if transform is None:
                continue
            
            # Extract translation
            translate = transform.find('.//translate')
            if translate is None:
                continue
            
            x = float(translate.get('x', 0))
            y = float(translate.get('y', 0))
            z = float(translate.get('z', 0))
            position = [x, y, z]
            
            # Extract scale
            scale = transform.find('.//scale')
            if scale is None:
                continue
            
            scale_x = float(scale.get('x', 1))
            scale_y = float(scale.get('y', 1))
            scale_z = float(scale.get('z', 1))
            
            # Check for rotation (walls are rotated 90 degrees around x-axis)
            rotate = transform.find('.//rotate')
            is_rotated = rotate is not None
            
            # Determine actual dimensions based on rotation
            # In Sionna XML, walls are rectangles rotated 90 degrees around x-axis
            # For top-down 2D view, we need to extract the horizontal dimensions
            if is_rotated:
                # Wall rotated 90 degrees: scale_x becomes length, scale_y becomes height (vertical), scale_z becomes width (thickness)
                # For 2D top-down view: length = scale_x*2, width = scale_z*2 (thickness)
                length = scale_x * 2  # Horizontal length along wall
                width = scale_z * 2   # Wall thickness (perpendicular)
                height = scale_y * 2  # Vertical height (not used in 2D view)
            else:
                # Floor/ceiling: horizontal, not rotated
                length = scale_x * 2
                width = scale_y * 2
                height = 0.1  # Default thin for visualization
            
            size = [length, width, height]
            
            # Categorize by shape name/type
            if 'wall' in shape_name.lower():
                geometry['walls'].append({
                    'position': position,
                    'size': size,
                    'material': material,
                    'name': shape_name
                })
            elif 'floor' in shape_name.lower():
                geometry['floors'].append({
                    'position': position,
                    'size': size,
                    'material': material,
                    'name': shape_name
                })
            elif 'ceiling' in shape_name.lower():
                geometry['ceilings'].append({
                    'position': position,
                    'size': size,
                    'material': material,
                    'name': shape_name
                })
            elif 'door' in shape_name.lower():
                geometry['doors'].append({
                    'position': position,
                    'size': size,
                    'material': material,
                    'name': shape_name
                })
            elif 'window' in shape_name.lower():
                geometry['windows'].append({
                    'position': position,
                    'size': size,
                    'material': material,
                    'name': shape_name
                })
            else:
                # Other obstacles
                geometry['obstacles'].append({
                    'position': position,
                    'size': size,
                    'material': material,
                    'name': shape_name
                })
        
        return geometry, materials
    except Exception as e:
        print(f"Error parsing XML scene: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
    
    def __init__(self, scene_file, frequency_mhz=5200, tx_power_dbm=20.0):
        self.scene_file = scene_file
        self.frequency_hz = frequency_mhz * 1e6
        self.scene = None
        self.tx_power_dbm = tx_power_dbm
        
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
    
    def get_receiver_positions(self):
        """Get all receiver positions from the scene"""
        if self.scene is None:
            return None
        positions = []
        for rx in self.scene.receivers:
            positions.append(rx.position.numpy() if hasattr(rx.position, 'numpy') else rx.position)
        return np.array(positions)
    
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
            channel_powers = []  # For debugging
            # Handle different possible shapes: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
            if len(a_complex.shape) == 6:
                num_rx = a_complex.shape[0]
                for rx_idx in range(num_rx):
                    # Extract path gains for this receiver (assuming single TX, single antenna)
                    path_gains = a_complex[rx_idx, 0, 0, 0, :, 0]  # [num_paths]
                    channel_power = np.sum(np.abs(path_gains) ** 2)
                    channel_powers.append(channel_power)
                    
                    # Use configured transmit power
                    # Note: Real WiFi APs typically transmit at 15-20 dBm, but Sionna simulations
                    # often need lower effective TX power to account for:
                    # - Simplified building geometry (missing furniture, equipment, people)
                    # - Idealized antenna patterns (real antennas aren't perfectly isotropic)
                    # - Material property uncertainties
                    # - Multi-floor effects not fully captured
                    tx_power_dbm = self.tx_power_dbm
                    if channel_power > 0:
                        path_loss_db = -10 * np.log10(channel_power)
                        rssi = tx_power_dbm - path_loss_db
                    else:
                        rssi = -100.0  # Very weak signal
                    
                    rssi_values.append(float(rssi))
                
                # Debug output
                if len(channel_powers) > 0:
                    avg_channel_power = np.mean(channel_powers)
                    print(f"   Channel powers: min={min(channel_powers):.2e}, max={max(channel_powers):.2e}, avg={avg_channel_power:.2e}")
                    print(f"   Simulated RSSI range: {min(rssi_values):.1f} to {max(rssi_values):.1f} dBm")
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
            # Return both RSSI values and paths object for visualization
            return np.array(rssi_values), paths
            
        except Exception as e:
            print(f"Ray tracing failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def generate_coverage_heatmap(self, building_name, rx_positions, measured_rssi, output_dir='results'):
        """
        Generate a coverage heatmap using scene.coverage_map and overlay measured points.
        
        Args:
            building_name: Name of the building
            rx_positions: Array of receiver positions [N, 3]
            measured_rssi: Array of measured RSSI values [N]
            output_dir: Directory to save the heatmap
        """
        if self.scene is None:
            print(f"⚠ Cannot generate heatmap for {building_name}: scene not loaded")
            return None
        
        try:
            # Compute coverage map
            print(f"Computing coverage map for {building_name}...")
            try:
                coverage_map = self.scene.coverage_map(
                    num_samples=int(1e6),
                    max_depth=3,
                    los=True,
                    reflection=True,
                    diffraction=True
                )
            except AttributeError:
                print(f"⚠ Native coverage_map not supported by this Sionna version. Skipping heatmap.")
                return None
            
            # Extract coverage data
            # coverage_map typically returns a CoverageMap object with power_map attribute
            if hasattr(coverage_map, 'power_map'):
                power_map = coverage_map.power_map
            elif hasattr(coverage_map, 'rssi'):
                power_map = coverage_map.rssi
            else:
                # Try to get it as a numpy array directly
                power_map = np.array(coverage_map) if hasattr(coverage_map, '__array__') else None
            
            if power_map is None:
                print(f"⚠ Could not extract power map from coverage_map for {building_name}")
                return None
            
            # Convert power map to RSSI (dBm)
            if hasattr(power_map, 'numpy'):
                power_map_np = power_map.numpy()
            else:
                power_map_np = np.array(power_map)
            
            # If power_map is in linear scale, convert to dBm
            if np.any(power_map_np > 0):
                # Check if it's already in dB scale (negative values typical)
                if np.all(power_map_np <= 0) or np.any(power_map_np < -50):
                    rssi_map = power_map_np  # Already in dB
                else:
                    # Convert from linear power to dBm
                    tx_power_dbm = 20.0
                    rssi_map = tx_power_dbm - 10 * np.log10(power_map_np + 1e-10)
            else:
                rssi_map = power_map_np
            
            # Get the grid coordinates from coverage_map
            if hasattr(coverage_map, 'x_min') and hasattr(coverage_map, 'x_max'):
                x_min, x_max = coverage_map.x_min, coverage_map.x_max
                y_min, y_max = coverage_map.y_min, coverage_map.y_max
            elif hasattr(coverage_map, 'bounding_box'):
                bbox = coverage_map.bounding_box
                x_min, y_min = bbox[0]
                x_max, y_max = bbox[1]
            else:
                # Fallback: estimate from scene bounds
                # Try to get scene bounds
                try:
                    from sionna.rt import Scene
                    if hasattr(self.scene, 'bounding_box'):
                        bbox = self.scene.bounding_box
                        x_min, y_min = bbox[0][:2]
                        x_max, y_max = bbox[1][:2]
                    else:
                        # Use receiver positions to estimate bounds
                        x_min, x_max = rx_positions[:, 0].min() - 10, rx_positions[:, 0].max() + 10
                        y_min, y_max = rx_positions[:, 1].min() - 10, rx_positions[:, 1].max() + 10
                except:
                    x_min, x_max = rx_positions[:, 0].min() - 10, rx_positions[:, 0].max() + 10
                    y_min, y_max = rx_positions[:, 1].min() - 10, rx_positions[:, 1].max() + 10
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))
            
            # Plot heatmap
            if len(rssi_map.shape) == 2:
                # 2D heatmap
                im = ax.imshow(
                    rssi_map, 
                    extent=[x_min, x_max, y_min, y_max],
                    origin='lower',
                    cmap='RdYlGn_r',
                    aspect='auto',
                    interpolation='bilinear'
                )
            elif len(rssi_map.shape) == 3:
                # 3D array - take a slice (e.g., middle z-slice or average)
                rssi_2d = np.mean(rssi_map, axis=2) if rssi_map.shape[2] > 1 else rssi_map[:, :, 0]
                im = ax.imshow(
                    rssi_2d,
                    extent=[x_min, x_max, y_min, y_max],
                    origin='lower',
                    cmap='RdYlGn_r',
                    aspect='auto',
                    interpolation='bilinear'
                )
            else:
                print(f"⚠ Unexpected coverage map shape: {rssi_map.shape}")
                return None
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Simulated RSSI (dBm)', fontsize=12)
            
            # Overlay measured points
            scatter = ax.scatter(
                rx_positions[:, 0], 
                rx_positions[:, 1],
                c=measured_rssi,
                s=300,
                marker='o',
                edgecolors='black',
                linewidths=2,
                cmap='RdYlGn_r',
                vmin=rssi_map.min() if hasattr(rssi_map, 'min') else -100,
                vmax=rssi_map.max() if hasattr(rssi_map, 'max') else -30,
                zorder=5,
                label='Measured Points'
            )
            
            # Annotate measured points with RSSI values
            for i, (pos, rssi) in enumerate(zip(rx_positions, measured_rssi)):
                ax.annotate(
                    f'{rssi:.0f}',
                    (pos[0], pos[1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    zorder=6
                )
            
            ax.set_xlabel('X Position (meters)', fontsize=12)
            ax.set_ylabel('Y Position (meters)', fontsize=12)
            ax.set_title(f'{building_name} - Coverage Heatmap with Measured Points Overlay', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # Save figure
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(
                output_dir,
                f"{building_name.replace(' ', '_')}_coverage_heatmap.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved coverage heatmap: {filename}")
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"⚠ Error generating heatmap for {building_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

class SimulationRunner:
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.results = {}
        
        os.makedirs(config['results_dir'], exist_ok=True)
    
    def validate_receiver_positions(self, building_name, rx_positions, scene_file=None):
        """
        Check if receiver positions are inside building boundaries.
        Returns list of indices that are outside boundaries.
        """
        if scene_file is None or not os.path.exists(scene_file):
            print(f"⚠ Cannot validate receiver positions for {building_name}: scene file not available")
            return []
        
        geometry, _ = parse_scene_xml(scene_file)
        if geometry is None:
            print(f"⚠ Cannot validate receiver positions for {building_name}: failed to parse geometry")
            return []
        
        # Extract building boundaries from floors (assuming floors define the building footprint)
        if not geometry['floors']:
            print(f"⚠ Cannot validate receiver positions for {building_name}: no floor geometry found")
            return []
        
        # Calculate bounding box from all floors
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for floor in geometry['floors']:
            pos = floor['position']
            size = floor['size']
            min_x = min(min_x, pos[0] - size[0]/2)
            max_x = max(max_x, pos[0] + size[0]/2)
            min_y = min(min_y, pos[1] - size[1]/2)
            max_y = max(max_y, pos[1] + size[1]/2)
        
        # Check each receiver position
        outside_indices = []
        for i, rx_pos in enumerate(rx_positions):
            x, y = rx_pos[0], rx_pos[1]
            if x < min_x or x > max_x or y < min_y or y > max_y:
                outside_indices.append(i)
        
        if outside_indices:
            print(f"⚠ {building_name}: {len(outside_indices)} receiver(s) outside building boundaries")
            for idx in outside_indices:
                rx_pos = rx_positions[idx]
                print(f"   Receiver {idx} at [{rx_pos[0]:.2f}, {rx_pos[1]:.2f}] is outside bounds "
                      f"[{min_x:.2f}, {max_x:.2f}] x [{min_y:.2f}, {max_y:.2f}]")
        else:
            print(f"✓ {building_name}: All receiver positions are inside building boundaries")
        
        return outside_indices
    
    def validate_building_dimensions(self, building_name, scene_file=None, csv_area_sqft=None):
        """
        Compare calculated building area from scene geometry vs CSV 'Actual' sqft.
        Flags discrepancies >10%.
        Returns (calculated_area_sqft, csv_area_sqft, discrepancy_percent)
        """
        if scene_file is None or not os.path.exists(scene_file):
            print(f"⚠ Cannot validate dimensions for {building_name}: scene file not available")
            return None, csv_area_sqft, None
        
        geometry, _ = parse_scene_xml(scene_file)
        if geometry is None:
            print(f"⚠ Cannot validate dimensions for {building_name}: failed to parse geometry")
            return None, csv_area_sqft, None
        
        # Calculate total area from floors
        total_area_sqft = 0.0
        for floor in geometry['floors']:
            pos = floor['position']
            size = floor['size']
            # Convert from m² to ft²
            area_m2 = size[0] * size[1]
            area_ft2 = area_m2 * 10.764  # 1 m² = 10.764 ft²
            total_area_sqft += area_ft2
        
        if csv_area_sqft is None or pd.isna(csv_area_sqft):
            print(f"⚠ {building_name}: CSV 'Actual' area not available for comparison")
            print(f"   Calculated area: {total_area_sqft:.0f} sq ft")
            return total_area_sqft, None, None
        
        # Calculate discrepancy
        try:
            csv_area = float(csv_area_sqft)
            discrepancy = abs(total_area_sqft - csv_area) / csv_area * 100
            
            if discrepancy > 10.0:
                print(f"⚠ {building_name}: Area discrepancy {discrepancy:.1f}% (>10% threshold)")
                print(f"   Calculated: {total_area_sqft:.0f} sq ft, CSV 'Actual': {csv_area:.0f} sq ft")
                print(f"   This may indicate multi-floor building or geometry mismatch")
            else:
                print(f"✓ {building_name}: Area matches CSV (discrepancy: {discrepancy:.1f}%)")
                print(f"   Calculated: {total_area_sqft:.0f} sq ft, CSV 'Actual': {csv_area:.0f} sq ft")
            
            return total_area_sqft, csv_area, discrepancy
        except (ValueError, TypeError):
            print(f"⚠ {building_name}: Could not parse CSV area value: {csv_area_sqft}")
            return total_area_sqft, csv_area_sqft, None
    
    def check_measurement_location_coverage(self, data):
        """
        Report which CSV locations are missing from MEASUREMENT_LOCATIONS dictionary.
        Returns dictionary mapping building names to lists of missing locations.
        """
        missing = {}
        
        for building_name in data['Hall'].unique():
            building_data = data[data['Hall'] == building_name]
            locations = building_data['Location'].unique()
            
            if building_name not in MEASUREMENT_LOCATIONS:
                missing[building_name] = list(locations)
                continue
            
            building_locations = MEASUREMENT_LOCATIONS[building_name]
            missing_locs = []
            
            for loc in locations:
                # Check exact match
                if loc not in building_locations:
                    # Check case-insensitive match
                    loc_lower = loc.lower()
                    found = False
                    for key in building_locations.keys():
                        if key.lower() == loc_lower:
                            found = True
                            break
                    if not found:
                        missing_locs.append(loc)
            
            if missing_locs:
                missing[building_name] = missing_locs
        
        if missing:
            print(f"\n⚠ Missing measurement locations in MEASUREMENT_LOCATIONS:")
            for building, locs in missing.items():
                print(f"   {building}: {locs}")
                if building in MEASUREMENT_LOCATIONS:
                    print(f"      Available: {list(MEASUREMENT_LOCATIONS[building].keys())}")
        else:
            print(f"\n✓ All measurement locations are covered in MEASUREMENT_LOCATIONS")
        
        return missing
    
    def weighted_centroid_localization(self, rx_positions, rssi_values):
        weights = 10 ** (rssi_values / 10.0)
        weights = weights / np.sum(weights)
        weighted_positions = rx_positions * weights[:, np.newaxis]
        estimated_pos = np.sum(weighted_positions, axis=0)
        estimated_pos[2] = 3.5
        return estimated_pos
    
    def estimate_ap_location(self, building_data, rx_positions=None, rssi_values=None):
        building_name = building_data['Hall'].iloc[0] if len(building_data) > 0 else None
        
        # Check for manual AP position override
        if building_name and building_name in self.config.get('MANUAL_AP_POSITIONS', {}):
            manual_pos = self.config['MANUAL_AP_POSITIONS'][building_name]
            print(f"Using manual AP position for {building_name}: {manual_pos}")
            return manual_pos
        
        # Otherwise, use weighted centroid localization
        if rx_positions is None:
            rx_positions = self.estimate_rx_positions(building_data)
        if rssi_values is None:
            rssi_values = building_data['RSSI (dBm)'].values
        
        if len(rssi_values) > 0:
            return self.weighted_centroid_localization(rx_positions, rssi_values)
        else:
            return [0.0, 0.0, 3.0]
    
    def estimate_rx_positions(self, building_data):
        """
        Map 'Location' column from CSV to actual X/Y coordinates from MEASUREMENT_LOCATIONS dictionary.
        Returns real-world coordinates within each building's geometry.
        """
        building_name = building_data['Hall'].iloc[0] if len(building_data) > 0 else None
        locations = building_data['Location'].values
        positions = []
        missing_locations = []
        
        if building_name not in MEASUREMENT_LOCATIONS:
            print(f"⚠ Warning: Building '{building_name}' not found in MEASUREMENT_LOCATIONS")
            print(f"   Falling back to circular pattern for {len(locations)} locations")
            # Fallback to circular pattern if building not in dictionary
            for i, loc in enumerate(locations):
                angle = 2 * np.pi * i / len(locations)
                radius = 10 + (i % 3) * 5
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 1.5
                positions.append([x, y, z])
            return np.array(positions)
        
        building_locations = MEASUREMENT_LOCATIONS[building_name]
        
        for loc in locations:
            # Try exact match first
            if loc in building_locations:
                coords = building_locations[loc]
                positions.append([coords[0], coords[1], coords[2]])
            else:
                # Try case-insensitive match
                loc_lower = loc.lower()
                found = False
                for key, coords in building_locations.items():
                    if key.lower() == loc_lower:
                        positions.append([coords[0], coords[1], coords[2]])
                        found = True
                        break
                
                if not found:
                    missing_locations.append(loc)
                    # Use fallback: center of building with default height
                    print(f"⚠ Warning: Location '{loc}' not found in MEASUREMENT_LOCATIONS for {building_name}")
                    print(f"   Using fallback position [0, 0, 1.5]")
                    positions.append([0.0, 0.0, 1.5])
        
        if missing_locations:
            print(f"⚠ Missing locations for {building_name}: {missing_locations}")
            print(f"   Available locations: {list(building_locations.keys())}")
        
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
        
        # Check for mixed frequency bands (2.4 GHz vs 5 GHz)
        frequencies = building_data['Frequency (MHz)'].values
        freq_5ghz = frequencies >= 5000  # 5 GHz band
        freq_24ghz = frequencies < 3000  # 2.4 GHz band
        
        if np.any(freq_24ghz) and np.any(freq_5ghz):
            print(f"⚠ WARNING: {building_name} has MIXED 2.4 GHz and 5 GHz measurements!")
            print(f"   Frequencies: {sorted(set(frequencies))}")
            print(f"   Filtering to 5 GHz only (2.4/5 GHz can't be averaged!)")
            # Keep only 5 GHz measurements for simulation
            building_data = building_data[building_data['Frequency (MHz)'] >= 5000].copy()
            if len(building_data) == 0:
                print(f"   No 5 GHz measurements available - skipping {building_name}")
                return None
            print(f"   Remaining measurements after filtering: {len(building_data)}")
        
        measured_rssi = building_data['RSSI (dBm)'].values
        frequencies = building_data['Frequency (MHz)'].values
        avg_frequency = np.mean(frequencies)
        
        rx_positions = self.estimate_rx_positions(building_data)
        ap_position = self.estimate_ap_location(building_data, rx_positions, measured_rssi)
        
        print(f"AP position: {ap_position}")
        print(f"Average frequency: {avg_frequency:.0f} MHz")
        
        simulated_rssi = None
        method = "Unknown"
        scene_file = None
        tracer = None
        paths = None
        
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
                    tracer = SionnaRayTracer(
                        scene_file, 
                        avg_frequency,
                        tx_power_dbm=self.config.get('tx_power_dbm', 20.0)
                    )
                    if tracer.load_scene():
                        tracer.setup_transmitter(ap_position)
                        tracer.setup_receivers(rx_positions)
                        result = tracer.compute_coverage(max_depth=3, num_samples=5e5)
                        
                        if result is not None and result[0] is not None:
                            simulated_rssi, paths = result
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
        
        # Run validation checks
        print(f"\n--- Validation Checks for {building_name} ---")
        
        # Validate receiver positions
        self.validate_receiver_positions(building_name, rx_positions, scene_file)
        
        # Validate building dimensions (get CSV area if available)
        csv_area = None
        if 'Actual' in building_data.columns:
            csv_area_values = building_data['Actual'].dropna().unique()
            if len(csv_area_values) > 0:
                # Try to extract numeric value (handle strings like "182,080 sq ft")
                try:
                    csv_area_str = str(csv_area_values[0]).replace(',', '').replace(' sq ft', '').strip()
                    csv_area = float(csv_area_str)
                except (ValueError, AttributeError):
                    pass
        
        self.validate_building_dimensions(building_name, scene_file, csv_area)
        
        # Apply per-building calibration offset
        # The simulation gives good relative patterns (high correlation) but may have
        # systematic offsets due to simplified geometry. We calibrate to match mean measured RSSI.
        if self.config.get('auto_calibrate', True) and len(simulated_rssi) > 0:
            mean_measured = np.mean(measured_rssi)
            mean_simulated = np.mean(simulated_rssi)
            calibration_offset = mean_measured - mean_simulated
            
            print(f"\n📊 Auto-calibration for {building_name}:")
            print(f"   Mean measured RSSI: {mean_measured:.1f} dBm")
            print(f"   Mean simulated RSSI: {mean_simulated:.1f} dBm")
            print(f"   Applying offset: {calibration_offset:+.1f} dB")
            
            simulated_rssi = simulated_rssi + calibration_offset
        
        metrics = self.calculate_metrics(simulated_rssi, measured_rssi)
        
        # Debug: Print actual values for buildings with high error
        if metrics['MAE'] > 15.0:
            print(f"\n⚠ High error detected for {building_name} - Showing detailed comparison:")
            print(f"   Location                 | Measured RSSI | Simulated RSSI | Error")
            print(f"   {'-'*70}")
            for i, (loc, meas, sim) in enumerate(zip(building_data['Location'].values, measured_rssi, simulated_rssi)):
                error = sim - meas
                print(f"   {loc[:25]:<25} | {meas:>7.1f} dBm   | {sim:>8.1f} dBm   | {error:>+6.1f} dB")
        
        result = {
            'building': building_name,
            'method': method,
            'measured_rssi': measured_rssi,
            'simulated_rssi': simulated_rssi,
            'locations': building_data['Location'].values,
            'metrics': metrics,
            'ap_position': ap_position,
            'rx_positions': rx_positions,
            'scene_file': scene_file,
            'tracer': tracer,
            'paths': paths
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
            self._plot_validation_overlay(result)
            
            # Generate coverage heatmap if Sionna ray tracer is available
            tracer = result.get('tracer')
            if tracer is not None and tracer.scene is not None:
                tracer.generate_coverage_heatmap(
                    building,
                    result['rx_positions'],
                    result['measured_rssi'],
                    self.config['results_dir']
                )
        
        self._plot_summary()
    
    def _plot_validation_overlay(self, result):
        """
        Create a top-down 2D validation view showing:
        - Building walls/boundaries
        - Receiver dots labeled with location names
        - AP position as red triangle
        - Distance circles (10m, 20m, 30m)
        - Color-coded materials
        """
        building = result['building']
        ap_position = result['ap_position']
        rx_positions = result['rx_positions']
        locations = result.get('locations', [])
        scene_file = result.get('scene_file')
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Material color mapping
        material_colors = {
            'concrete': '#8B7355',
            'brick': '#A0522D',
            'drywall': '#F5F5DC',
            'glass': '#87CEEB',
            'wood': '#DEB887',
            'plasterboard': '#F5F5DC',
            'metal': '#708090'
        }
        
        material_alphas = {
            'concrete': 0.8,
            'brick': 0.8,
            'drywall': 0.6,
            'glass': 0.3,
            'wood': 0.7,
            'plasterboard': 0.6,
            'metal': 0.8
        }
        
        # Parse and draw building geometry
        geometry = None
        if scene_file and os.path.exists(scene_file):
            geometry, materials = parse_scene_xml(scene_file)
            
            if geometry:
                # Draw floors (as background)
                for floor in geometry['floors']:
                    pos = floor['position']
                    size = floor['size']
                    mat = floor['material']
                    color = material_colors.get(mat, '#CCCCCC')
                    alpha = material_alphas.get(mat, 0.5)
                    
                    rect = Rectangle(
                        (pos[0] - size[0]/2, pos[1] - size[1]/2),
                        size[0], size[1],
                        facecolor=color, edgecolor='black', linewidth=0.5,
                        alpha=alpha, zorder=0
                    )
                    ax.add_patch(rect)
                
                # Draw walls
                for wall in geometry['walls']:
                    pos = wall['position']
                    size = wall['size']
                    mat = wall['material']
                    color = material_colors.get(mat, '#A0522D')
                    alpha = material_alphas.get(mat, 0.8)
                    
                    wall_thickness = max(size[1], 0.3)
                    rect = Rectangle(
                        (pos[0] - size[0]/2, pos[1] - wall_thickness/2),
                        size[0], wall_thickness,
                        facecolor=color, edgecolor='black', linewidth=1,
                        alpha=alpha, zorder=2
                    )
                    ax.add_patch(rect)
        
        # Draw distance circles from AP
        for radius in [10, 20, 30]:
            circle = Circle(
                (ap_position[0], ap_position[1]), radius,
                fill=False, edgecolor='gray', linestyle='--',
                linewidth=1, alpha=0.5, zorder=1
            )
            ax.add_patch(circle)
            # Label the circle
            ax.text(ap_position[0] + radius, ap_position[1], f'{radius}m',
                   fontsize=8, color='gray', alpha=0.7, ha='left', va='center')
        
        # Draw access point (red triangle)
        ax.scatter(ap_position[0], ap_position[1], s=500, marker='^', 
                  color='red', label='Access Point', zorder=5, 
                  edgecolors='black', linewidths=2)
        
        # Draw receiver points with location labels
        for i, (pos, loc) in enumerate(zip(rx_positions, locations)):
            ax.scatter(pos[0], pos[1], s=200, marker='o', 
                      color='blue', edgecolors='black', linewidths=1.5, 
                      zorder=4, alpha=0.7)
            # Label with location name
            ax.annotate(loc, (pos[0], pos[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', alpha=0.8),
                       zorder=6)
        
        # Add material legend
        if geometry:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=material_colors.get('brick', '#A0522D'), alpha=0.8, label='Brick'),
                Patch(facecolor=material_colors.get('drywall', '#F5F5DC'), alpha=0.6, label='Drywall'),
                Patch(facecolor=material_colors.get('glass', '#87CEEB'), alpha=0.3, label='Glass'),
                Patch(facecolor=material_colors.get('wood', '#DEB887'), alpha=0.6, label='Wood'),
                Patch(facecolor=material_colors.get('concrete', '#8B7355'), alpha=0.8, label='Concrete'),
                Patch(facecolor=material_colors.get('metal', '#708090'), alpha=0.8, label='Metal'),
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=9, title='Materials')
        
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title(f'{building} - Validation Overlay: Building Layout, Receivers, and AP', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        filename = os.path.join(
            self.config['results_dir'],
            f"{building.replace(' ', '_')}_validation_overlay.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved validation overlay: {filename}")
        plt.close()
    
    def _plot_scene_layout(self, result):
        building = result['building']
        ap_position = result['ap_position']
        rx_positions = result['rx_positions']
        measured_rssi = result['measured_rssi']
        scene_file = result.get('scene_file')
        tracer = result.get('tracer')
        paths = result.get('paths')
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Material color mapping
        material_colors = {
            'concrete': '#8B7355',  # Brown-gray
            'brick': '#A0522D',     # Sienna
            'drywall': '#F5F5DC',   # Beige
            'glass': '#87CEEB',     # Sky blue (semi-transparent)
            'wood': '#DEB887',      # Burlywood
            'plasterboard': '#F5F5DC',  # Beige (same as drywall)
            'metal': '#708090'      # Slate gray
        }
        
        material_alphas = {
            'concrete': 0.8,
            'brick': 0.8,
            'drywall': 0.6,
            'glass': 0.3,
            'wood': 0.7,
            'plasterboard': 0.6,
            'metal': 0.8
        }
        
        # Parse and draw building geometry
        geometry = None
        if scene_file and os.path.exists(scene_file):
            geometry, materials = parse_scene_xml(scene_file)
            
            if geometry:
                # Draw floors (as background)
                for floor in geometry['floors']:
                    pos = floor['position']
                    size = floor['size']
                    mat = floor['material']
                    color = material_colors.get(mat, '#CCCCCC')
                    alpha = material_alphas.get(mat, 0.5)
                    
                    rect = Rectangle(
                        (pos[0] - size[0]/2, pos[1] - size[1]/2),
                        size[0], size[1],
                        facecolor=color, edgecolor='black', linewidth=0.5,
                        alpha=alpha, zorder=0, label='Floor' if floor == geometry['floors'][0] else ''
                    )
                    ax.add_patch(rect)
                
                # Draw walls
                for wall in geometry['walls']:
                    pos = wall['position']
                    size = wall['size']
                    mat = wall['material']
                    color = material_colors.get(mat, '#A0522D')
                    alpha = material_alphas.get(mat, 0.8)
                    
                    # For top-down 2D view: size[0] = length, size[1] = thickness
                    # Walls are drawn as rectangles with their length and thickness
                    # Use minimum thickness of 0.3m for visibility if too thin
                    wall_thickness = max(size[1], 0.3)
                    
                    rect = Rectangle(
                        (pos[0] - size[0]/2, pos[1] - wall_thickness/2),
                        size[0], wall_thickness,
                        facecolor=color, edgecolor='black', linewidth=1,
                        alpha=alpha, zorder=2, label='Wall' if wall == geometry['walls'][0] else ''
                    )
                    ax.add_patch(rect)
                
                # Draw doors
                for door in geometry['doors']:
                    pos = door['position']
                    size = door['size']
                    mat = door['material']
                    color = material_colors.get(mat, '#DEB887')
                    
                    if size[0] > size[1]:  # Horizontal door
                        rect = Rectangle(
                            (pos[0] - size[0]/2, pos[1] - size[1]/2),
                            size[0], size[1],
                            facecolor=color, edgecolor='brown', linewidth=1.5,
                            alpha=0.6, zorder=2, linestyle='--', label='Door' if door == geometry['doors'][0] else ''
                        )
                    else:  # Vertical door
                        rect = Rectangle(
                            (pos[0] - size[1]/2, pos[1] - size[0]/2),
                            size[1], size[0],
                            facecolor=color, edgecolor='brown', linewidth=1.5,
                            alpha=0.6, zorder=2, linestyle='--', label='Door' if door == geometry['doors'][0] else ''
                        )
                    ax.add_patch(rect)
                
                # Draw windows
                for window in geometry['windows']:
                    pos = window['position']
                    size = window['size']
                    mat = window['material']
                    color = material_colors.get(mat, '#87CEEB')
                    
                    if size[0] > size[1]:  # Horizontal window
                        rect = Rectangle(
                            (pos[0] - size[0]/2, pos[1] - size[1]/2),
                            size[0], size[1],
                            facecolor=color, edgecolor='blue', linewidth=1,
                            alpha=0.3, zorder=2, label='Window' if window == geometry['windows'][0] else ''
                        )
                    else:  # Vertical window
                        rect = Rectangle(
                            (pos[0] - size[1]/2, pos[1] - size[0]/2),
                            size[1], size[0],
                            facecolor=color, edgecolor='blue', linewidth=1,
                            alpha=0.3, zorder=2, label='Window' if window == geometry['windows'][0] else ''
                        )
                    ax.add_patch(rect)
                
                # Draw obstacles (columns, etc.)
                for obstacle in geometry['obstacles']:
                    pos = obstacle['position']
                    size = obstacle['size']
                    mat = obstacle['material']
                    color = material_colors.get(mat, '#708090')
                    
                    rect = Rectangle(
                        (pos[0] - size[0]/2, pos[1] - size[1]/2),
                        size[0], size[1],
                        facecolor=color, edgecolor='black', linewidth=1,
                        alpha=0.7, zorder=2, label='Obstacle' if obstacle == geometry['obstacles'][0] else ''
                    )
                    ax.add_patch(rect)
        
        # Draw ray paths if available
        if paths is not None and tracer is not None:
            try:
                # Try to extract ray paths from paths object
                # This is a simplified visualization - showing strongest paths
                tx_pos = ap_position
                rx_positions_from_scene = tracer.get_receiver_positions()
                
                if rx_positions_from_scene is not None and len(rx_positions_from_scene) > 0:
                    # Draw ray paths (simplified - direct paths for now)
                    # In a full implementation, you'd extract actual reflection/diffraction paths
                    for rx_idx, rx_pos in enumerate(rx_positions_from_scene):
                        if len(rx_pos) >= 2:
                            # Draw direct path
                            ax.plot([tx_pos[0], rx_pos[0]], 
                                   [tx_pos[1], rx_pos[1]], 
                                   'b-', alpha=0.2, linewidth=0.5, zorder=1)
            except Exception as e:
                # Silently fail if ray path extraction doesn't work
                pass
        
        # Draw receiver grid from Sionna scene
        if tracer is not None and tracer.scene is not None:
            try:
                rx_grid_positions = tracer.get_receiver_positions()
                if rx_grid_positions is not None and len(rx_grid_positions) > 0:
                    # Draw receiver grid points
                    ax.scatter(rx_grid_positions[:, 0], rx_grid_positions[:, 1], 
                             s=50, marker='o', color='cyan', alpha=0.4,
                             edgecolors='blue', linewidths=0.5, zorder=3,
                             label='Receiver Grid')
            except Exception as e:
                pass
        
        # Draw access point
        ax.scatter(ap_position[0], ap_position[1], s=500, marker='^', 
                  color='red', label='Access Point', zorder=5, edgecolors='black', linewidths=2)
        
        # Draw measurement points with RSSI coloring
        scatter = ax.scatter(rx_positions[:, 0], rx_positions[:, 1], 
                           c=measured_rssi, s=200, cmap='RdYlGn_r', 
                           edgecolors='black', linewidths=1.5, zorder=4,
                           label='Measurement Points')
        
        # Annotate measurement points
        for i, (pos, rssi) in enumerate(zip(rx_positions, measured_rssi)):
            ax.annotate(f'{rssi:.0f} dBm', 
                       (pos[0], pos[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', alpha=0.8),
                       zorder=6)
        
        # Draw simple lines from AP to measurement points (if not showing ray paths)
        if paths is None:
            for rx_pos in rx_positions:
                ax.plot([ap_position[0], rx_pos[0]], 
                       [ap_position[1], rx_pos[1]], 
                       'k--', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Add colorbar for RSSI
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Measured RSSI (dBm)', fontsize=12)
        
        # Add material legend if geometry was drawn
        if geometry:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=material_colors.get('brick', '#A0522D'), alpha=0.8, label='Brick'),
                Patch(facecolor=material_colors.get('drywall', '#F5F5DC'), alpha=0.6, label='Drywall'),
                Patch(facecolor=material_colors.get('glass', '#87CEEB'), alpha=0.3, label='Glass'),
                Patch(facecolor=material_colors.get('wood', '#DEB887'), alpha=0.6, label='Wood'),
                Patch(facecolor=material_colors.get('concrete', '#8B7355'), alpha=0.8, label='Concrete'),
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=9, title='Materials')
        
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title(f'{building} - Scene Layout with Building Geometry, Materials, Receiver Grid, and Ray Paths', 
                    fontsize=14, fontweight='bold')
        
        # Add main legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
        
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
    VALID_BUILDINGS = {
        'Kiewit', 'Kauffman', 'Adele Coryell', 'Love Library South', 
        'Selleck', 'Brace', 'Hamilton', 'Bessey', 'Union', 
        'Oldfather', 'Burnett', 'Memorial Stadium'
    }
    
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
    
    # Run validation checks after loading data
    print(f"\n{'='*70}")
    print("RUNNING DATA VALIDATION")
    print(f"{'='*70}")
    
    runner = SimulationRunner(data, CONFIG)
    
    # Check measurement location coverage
    missing_locations = runner.check_measurement_location_coverage(data)
    
    print(f"\n{'='*70}")
    print("STARTING SIMULATIONS")
    print(f"{'='*70}")
    
    results = runner.run_all_buildings()
    
    runner.create_visualizations()
    
    runner.print_summary()
    
    print(f"\n{'-'*73}")
    print("SIMULATION COMPLETE!")


if __name__ == "__main__":
    main()