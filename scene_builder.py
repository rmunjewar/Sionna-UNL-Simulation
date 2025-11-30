import numpy as np
import os
import pandas as pd

class BuildingSceneBuilder:
    
    def __init__(self, building_name):
        self.building_name = building_name
        self.objects = []
        self.shape_counter = 0
        
    def add_floor(self, position, size, material='concrete'):
        obj = {'type': 'floor', 'position': position, 'size': size, 'material': material}
        self.objects.append(obj)
        
    def add_ceiling(self, position, size, material='concrete'):
        obj = {'type': 'ceiling', 'position': position, 'size': size, 'material': material}
        self.objects.append(obj)
        
    def add_wall(self, position, size, material='brick'):
        obj = {'type': 'wall', 'position': position, 'size': size, 'material': material}
        self.objects.append(obj)
        
    def add_door(self, position, size, material='wood'):
        obj = {'type': 'door', 'position': position, 'size': size, 'material': material}
        self.objects.append(obj)
        
    def add_window(self, position, size, material='glass'):
        obj = {'type': 'window', 'position': position, 'size': size, 'material': material}
        self.objects.append(obj)
        
    def add_room(self, corner_position, dimensions, height, wall_material='drywall', door_width=1.0):
        x, y, z = corner_position
        length, width = dimensions
        wall_thick = 0.15
        
        # Add walls with a door opening in one wall
        # Front wall (positive Y) with door
        wall_length = length
        door_gap = door_width
        if wall_length > door_gap:
            # Split wall into two segments with door in middle
            left_seg = (wall_length - door_gap) / 2
            self.add_wall([x + left_seg/2, y + width, z + height/2], [left_seg, wall_thick, height], wall_material)
            self.add_wall([x + length - left_seg/2, y + width, z + height/2], [left_seg, wall_thick, height], wall_material)
            # Add door in the gap
            self.add_door([x + length/2, y + width, z + height/2], [door_gap, wall_thick, height*0.8], 'wood')
        else:
            self.add_wall([x + length/2, y + width, z + height/2], [length, wall_thick, height], wall_material)
        
        # Back wall (negative Y)
        self.add_wall([x + length/2, y, z + height/2], [length, wall_thick, height], wall_material)
        # Right wall (positive X)
        self.add_wall([x + length, y + width/2, z + height/2], [wall_thick, width, height], wall_material)
        # Left wall (negative X)
        self.add_wall([x, y + width/2, z + height/2], [wall_thick, width, height], wall_material)

    def generate_xml(self, output_dir='scenes'):
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{self.building_name}.xml')
        xml_content = self._build_header() + self._build_materials() + "".join([self._obj_to_xml(o) for o in self.objects]) + self._build_footer()
        with open(filename, 'w') as f:
            f.write(xml_content)
        print(f"o Generated: {filename}")
        return filename
    
    def _build_header(self):
        return """<?xml version="1.0" encoding="utf-8"?>
<scene version="2.1.0">

"""
    
    def _build_materials(self):
        """Build material definitions for all unique materials used"""
        unique_materials = set(obj['material'] for obj in self.objects)
        materials_xml = "<!-- Materials -->\n\n"
        
        for mat_name in sorted(unique_materials):
            mat_id = f"mat-{mat_name}"
            mat_def = self._get_mat_definition(mat_name, mat_id)
            materials_xml += f"\t{mat_def}\n"
        
        materials_xml += "\n<!-- Shapes -->\n\n"
        return materials_xml
    
    def _build_footer(self):
        return "</scene>"
    
    def _obj_to_xml(self, obj):
        mat_id = f"mat-{obj['material']}"
        pos, size = obj['position'], obj['size']
        self.shape_counter += 1
        shape_id = f"shape-{obj['type']}-{self.shape_counter}"
        shape_name = f"{obj['type']}_{self.shape_counter}"
        
        if obj['type'] in ['floor', 'ceiling']:
            return f"""\t<shape type="rectangle" id="{shape_id}" name="{shape_name}">
\t\t<transform name="to_world">
\t\t\t<translate x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>
\t\t\t<scale x="{size[0]/2}" y="{size[1]/2}" z="1"/>
\t\t</transform>
\t\t<ref id="{mat_id}" name="bsdf"/>
\t</shape>
"""
        else: 
            return f"""\t<shape type="rectangle" id="{shape_id}" name="{shape_name}">
\t\t<transform name="to_world">
\t\t\t<translate x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>
\t\t\t<rotate x="1" angle="90"/>
\t\t\t<scale x="{size[0]/2}" y="{size[2]/2}" z="1"/>
\t\t</transform>
\t\t<ref id="{mat_id}" name="bsdf"/>
\t</shape>
"""

    def _get_mat_definition(self, name, mat_id):
        """Get material definition with ID for Sionna"""
        # Use Sionna-compatible ITU radio materials
        mats = {
            'concrete': '<bsdf type="itu-radio-material" id="{}"><string name="type" value="concrete"/><float name="thickness" value="0.1"/></bsdf>',
            'brick': '<bsdf type="itu-radio-material" id="{}"><string name="type" value="concrete"/><float name="thickness" value="0.2"/></bsdf>',
            'drywall': '<bsdf type="itu-radio-material" id="{}"><string name="type" value="plasterboard"/><float name="thickness" value="0.1"/></bsdf>',
            'glass': '<bsdf type="itu-radio-material" id="{}"><string name="type" value="glass"/><float name="thickness" value="0.01"/></bsdf>',
            'wood': '<bsdf type="itu-radio-material" id="{}"><string name="type" value="wood"/><float name="thickness" value="0.05"/></bsdf>',
            'metal': '<bsdf type="itu-radio-material" id="{}"><string name="type" value="metal"/><float name="thickness" value="0.1"/></bsdf>'
        }
        template = mats.get(name, mats['concrete'])
        return template.format(mat_id)

def ft_to_m(ft):
    return ft * 0.3048

def load_building_dimensions(csv_file='Wireless Communications - Data Collection - Data-2.csv'):
    try:
        try:
            df = pd.read_csv(csv_file, quotechar='"', escapechar='\\', on_bad_lines='skip')
        except Exception as e:
            print(f"⚠ Error reading CSV with default settings: {e}")
            df = pd.read_csv(csv_file, quotechar='"', on_bad_lines='skip', engine='python')
        
        df['Hall'] = df['Hall'].ffill()
        
        building_dims = {}
        for building in df['Hall'].dropna().unique():
            building_data = df[df['Hall'] == building]
            length = None
            width = None
            area = None
            floors = None
            
            length_vals = building_data['Length (ft)'].dropna()
            width_vals = building_data['Width (ft)'].dropna()
            area_vals = building_data['Area (sq ft)'].dropna()
            floors_vals = building_data['Floors'].dropna()
            
            if len(length_vals) > 0:
                length = float(length_vals.iloc[0])
            if len(width_vals) > 0:
                width = float(width_vals.iloc[0])
            if len(area_vals) > 0:
                area = float(area_vals.iloc[0])
            if len(floors_vals) > 0:
                try:
                    floors = int(floors_vals.iloc[0])
                except:
                    floors = None
            
            building_dims[building] = {
                'length_ft': length,
                'width_ft': width,
                'area_sqft': area,
                'floors': floors
            }
        
        # Validation: Check for expected buildings
        expected_buildings = ['Kiewit', 'Kauffman', 'Adele Coryell', 'Love Library South', 'Selleck', 'Brace']
        for building in expected_buildings:
            if building not in building_dims:
                print(f"⚠ Warning: {building} not found in CSV")
            elif not building_dims[building].get('length_ft') or not building_dims[building].get('width_ft'):
                print(f"⚠ Warning: {building} missing length or width")
        
        return building_dims
    except Exception as e:
        print(f"Could not load dimensions from CSV: {e}")
        return {}

def create_kiewit_scene(dims=None):
    scene = BuildingSceneBuilder('kiewit')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(126.38) 
        W = ft_to_m(245.73)
    
    # Add padding to floor/ceiling to prevent geometry leaks at edges
    padding = 1.0
    scene.add_floor([0,0,0], [L + padding, W + padding, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L + padding, W + padding, 0.1], 'drywall')
    
    scene.add_wall([0, W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    
    scene.add_room([10, 10, 0], [12, 10], 3.5, 'drywall')
    
    scene.add_wall([0, 0, 1.75], [1.0, 1.0, 3.5], 'concrete') 
    
    return scene

def create_kauffman_scene(dims=None):
    scene = BuildingSceneBuilder('kauffman')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(194.24)
        W = ft_to_m(201.42)
    
    # Add padding to floor/ceiling to prevent geometry leaks at edges
    padding = 1.0
    scene.add_floor([0,0,0], [L + padding, W + padding, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L + padding, W + padding, 0.1], 'drywall')
    
    # Exterior walls
    scene.add_wall([0, W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    
    # Internal rooms for chemistry building
    scene.add_room([-L/4, -W/4, 0], [L/3, W/3], 3.5, 'drywall')
    scene.add_room([L/4, W/4, 0], [L/3, W/3], 3.5, 'drywall')
    
    # Stairwell area
    scene.add_wall([0, 0, 1.75], [4.0, 4.0, 3.5], 'concrete')
    
    return scene

def create_adele_coryell_scene(dims=None):
    scene = BuildingSceneBuilder('adele_coryell')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(128.03)
        W = ft_to_m(252.56)
    
    # Single floor library/learning commons
    # Add padding to floor/ceiling to prevent geometry leaks at edges
    padding = 1.0
    scene.add_floor([0,0,0], [L + padding, W + padding, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L + padding, W + padding, 0.1], 'drywall')
    
    # Exterior walls
    scene.add_wall([0, W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    
    # Windows on perimeter
    scene.add_window([0, W/2, 1.75], [L*0.8, 0.1, 2.5], 'glass')
    scene.add_window([0, -W/2, 1.75], [L*0.8, 0.1, 2.5], 'glass')
    scene.add_window([L/2, 0, 1.75], [0.1, W*0.8, 2.5], 'glass')
    scene.add_window([-L/2, 0, 1.75], [0.1, W*0.8, 2.5], 'glass')
    
    # Few internal walls for open study areas
    scene.add_wall([0, 0, 1.75], [L*0.3, 0.15, 3.5], 'drywall')
    
    return scene

def create_love_library_south_scene(dims=None):
    scene = BuildingSceneBuilder('love_library_south')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(207.66)
        W = ft_to_m(151.56)
    
    # 3-floor library
    # Add padding to floor/ceiling to prevent geometry leaks at edges
    padding = 1.0
    scene.add_floor([0,0,0], [L + padding, W + padding, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L + padding, W + padding, 0.1], 'concrete')
    
    # Exterior walls
    scene.add_wall([0, W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    
    # Glass windows on exterior
    scene.add_window([0, W/2, 1.75], [L*0.9, 0.1, 2.5], 'glass')
    scene.add_window([0, -W/2, 1.75], [L*0.9, 0.1, 2.5], 'glass')
    scene.add_window([L/2, 0, 1.75], [0.1, W*0.9, 2.5], 'glass')
    scene.add_window([-L/2, 0, 1.75], [0.1, W*0.9, 2.5], 'glass')
    
    # Multiple internal support columns
    column_spacing = 15.0
    for x in np.arange(-L/3, L/3, column_spacing):
        for y in np.arange(-W/3, W/3, column_spacing):
            scene.add_wall([x, y, 1.75], [0.5, 0.5, 3.5], 'concrete')
    
    return scene

def create_selleck_scene(dims=None):
    scene = BuildingSceneBuilder('selleck')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(388.06)
        W = ft_to_m(207.22)
    
    # Add padding to floor/ceiling to prevent geometry leaks at edges
    padding = 1.0
    scene.add_floor([0,0,0], [L + padding, W + padding, 0.3], 'concrete')
    scene.add_ceiling([0,0,4.0], [L + padding, W + padding, 0.1], 'drywall')
    
    scene.add_wall([0, W/2, 2.0], [L, 0.4, 4.0], 'brick')
    
    return scene

def create_brace_scene(dims=None):
    scene = BuildingSceneBuilder('brace')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(220.32)
        W = ft_to_m(123.03)
    
    # Add padding to floor/ceiling to prevent geometry leaks at edges
    padding = 1.0
    scene.add_floor([0,0,0], [L + padding, W + padding, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L + padding, W + padding, 0.1], 'concrete')
    
    scene.add_room([10, 5, 0], [12, 10], 3.5, 'brick')
    
    return scene


def main():
    print("Generating scenes based on CSV data with accurate measurements...")
    
    building_dims = load_building_dimensions('Wireless Communications - Data Collection - Data-2.csv')
    
    building_map = {
        'Kiewit': (create_kiewit_scene, 'kiewit'),
        'Kauffman': (create_kauffman_scene, 'kauffman'),
        'Adele Coryell': (create_adele_coryell_scene, 'adele_coryell'),
        'Love Library South': (create_love_library_south_scene, 'love_library_south'),
        'Selleck': (create_selleck_scene, 'selleck'),
        'Brace': (create_brace_scene, 'brace')
    }
    
    for building_name, (builder_func, scene_name) in building_map.items():
        try:
            dims = building_dims.get(building_name, None)
            if dims:
                print(f"Using CSV dimensions for {building_name}: {dims.get('length_ft', 'N/A')} x {dims.get('width_ft', 'N/A')} ft")
            scene = builder_func(dims)
            scene.generate_xml()
        except Exception as e:
            print(f"Error generating {building_name}: {e}")

if __name__ == "__main__":
    main()