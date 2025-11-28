import numpy as np
import os
import pandas as pd

class BuildingSceneBuilder:
    
    def __init__(self, building_name):
        self.building_name = building_name
        self.objects = []
        
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
        
    def add_room(self, corner_position, dimensions, height, wall_material='drywall'):
        x, y, z = corner_position
        length, width = dimensions
        wall_thick = 0.15
        
        self.add_wall([x + length/2, y + width, z + height/2], [length, wall_thick, height], wall_material)
        self.add_wall([x + length/2, y, z + height/2], [length, wall_thick, height], wall_material)
        self.add_wall([x + length, y + width/2, z + height/2], [wall_thick, width, height], wall_material)
        self.add_wall([x, y + width/2, z + height/2], [wall_thick, width, height], wall_material)

    def generate_xml(self, output_dir='scenes'):
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{self.building_name}.xml')
        xml_content = self._build_header() + "".join([self._obj_to_xml(o) for o in self.objects]) + self._build_footer()
        with open(filename, 'w') as f:
            f.write(xml_content)
        print(f"o Generated: {filename}")
        return filename
    
    def _build_header(self):
        return """<?xml version="1.0" encoding="utf-8"?>
<scene version="0.6.0">
    <sensor type="perspective">
        <transform name="to_world">
            <lookat origin="0, 0, 10" target="0, 0, 0" up="0, 1, 0"/>
        </transform>
        <float name="fov" value="45"/>
    </sensor>
"""
    
    def _build_footer(self):
        return "</scene>"
    
    def _obj_to_xml(self, obj):
        mat = self._get_mat(obj['material'])
        pos, size = obj['position'], obj['size']
        
        if obj['type'] in ['floor', 'ceiling']:
            return f"""
    <shape type="rectangle">
        <transform name="to_world">
            <translate x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>
            <scale x="{size[0]/2}" y="{size[1]/2}" z="1"/>
        </transform>
        {mat}
    </shape>"""
        else: 
            return f"""
    <shape type="rectangle">
        <transform name="to_world">
            <translate x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>
            <rotate x="1" angle="90"/>
            <scale x="{size[0]/2}" y="{size[2]/2}" z="1"/>
        </transform>
        {mat}
    </shape>"""

    def _get_mat(self, name):
        mats = {
            'concrete': '<bsdf type="diffuse"><rgb name="reflectance" value="0.5, 0.5, 0.5"/></bsdf>',
            'brick': '<bsdf type="diffuse"><rgb name="reflectance" value="0.6, 0.4, 0.3"/></bsdf>',
            'drywall': '<bsdf type="diffuse"><rgb name="reflectance" value="0.8, 0.8, 0.75"/></bsdf>',
            'glass': '<bsdf type="dielectric"><float name="int_ior" value="1.5"/></bsdf>',
            'wood': '<bsdf type="diffuse"><rgb name="reflectance" value="0.6, 0.4, 0.2"/></bsdf>',
            'metal': '<bsdf type="conductor"><rgb name="specularReflectance" value="1, 1, 1"/></bsdf>'
        }
        return mats.get(name, mats['concrete'])

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
        
        if 'Love Library' in building_dims:
            if 'Adele Coryell' in building_dims and not building_dims['Adele Coryell'].get('length_ft'):
                building_dims['Adele Coryell'] = building_dims['Love Library'].copy()
            if 'Love Library South' in building_dims and not building_dims['Love Library South'].get('length_ft'):
                building_dims['Love Library South'] = building_dims['Love Library'].copy()
        
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
        L = ft_to_m(363.53) 
        W = ft_to_m(246.91)
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L, W, 0.1], 'drywall')
    
    scene.add_wall([0, W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [0.3, W, 3.5], 'brick')
    
    scene.add_room([10, 10, 0], [12, 10], 3.5, 'drywall')
    
    scene.add_wall([0, 0, 1.75], [1.0, 1.0, 3.5], 'concrete') 
    
    return scene

def create_kauffman_scene(dims=None):
    scene = BuildingSceneBuilder('kauffman')
    L, W = 60.0, 20.0 
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.0], [L, W, 0.1], 'drywall')
    
    scene.add_wall([0, 2.0, 1.5], [L, 0.15, 3.0], 'drywall') 
    scene.add_wall([0, -2.0, 1.5], [L, 0.15, 3.0], 'drywall') 
    
    scene.add_room([-20, 0, 0], [6, 4], 3.0, 'concrete')
    
    return scene

def create_bessey_scene(dims=None):
    scene = BuildingSceneBuilder('bessey')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(319.46)
        W = ft_to_m(239.09)
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L, W, 0.1], 'concrete')
    
    scene.add_wall([0, W/2, 1.75], [L, 0.5, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.5, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [0.5, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [0.5, W, 3.5], 'brick')
    
    return scene

def create_love_library_scene(dims=None):
    scene = BuildingSceneBuilder('love_library')
    
    L, W = 80.0, 60.0
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,4.5], [L, W, 0.1], 'concrete')
    
    scene.add_window([0, W/2, 2.25], [L, 0.1, 4.5], 'glass') 
    
    for x in [-20, 0, 20]:
        for y in [-20, 0, 20]:
            scene.add_wall([x, y, 2.25], [0.8, 0.8, 4.5], 'concrete')
            
    return scene

def create_edwards_scene(dims=None):
    scene = BuildingSceneBuilder('carolyn_pope_edwards')
    L, W = 100.0, 30.0
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,4.0], [L, W, 0.1], 'drywall')
    
    scene.add_window([0, W/2, 2.0], [L, 0.1, 4.0], 'glass')
    scene.add_window([0, -W/2, 2.0], [L, 0.1, 4.0], 'glass')
    
    return scene

def create_avery_scene(dims=None):
    scene = BuildingSceneBuilder('avery')
    L, W = 60.0, 40.0
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L, W, 0.1], 'concrete')
    
    scene.add_wall([0, W/2, 1.75], [L, 0.3, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, 0.3, 3.5], 'brick')
    
    scene.add_room([10, 0, 0], [10, 8], 3.5, 'drywall')
    
    return scene

def create_coliseum_scene(dims=None):
    scene = BuildingSceneBuilder('coliseum')
    L, W = 80.0, 50.0 
    H = 10.0
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'wood')
    scene.add_ceiling([0,0,H], [L, W, 0.5], 'metal')
    
    scene.add_wall([0, W/2, H/2], [L, 0.5, H], 'brick')
    scene.add_wall([0, -W/2, H/2], [L, 0.5, H], 'brick')
    scene.add_wall([L/2, 0, H/2], [0.5, W, H], 'brick')
    scene.add_wall([-L/2, 0, H/2], [0.5, W, H], 'brick')
    
    return scene

def create_union_scene(dims=None):
    scene = BuildingSceneBuilder('union')
    if dims and dims.get('area_sqft'):
        area_sqft = dims['area_sqft']
        side_ft = np.sqrt(area_sqft)
        L = ft_to_m(side_ft)
        W = ft_to_m(side_ft)
    else:
        L, W = 50.0, 50.0
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,4.0], [L, W, 0.1], 'drywall')
    
    scene.add_room([0, 0, 0], [4, 2], 4.0, 'brick') 
    
    return scene

def create_oldfather_scene(dims=None):
    scene = BuildingSceneBuilder('oldfather')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(126.55)
        W = ft_to_m(74.88)
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.0], [L, W, 0.1], 'concrete')
    
    scene.add_room([-2, 0, 0], [6, 6], 3.0, 'concrete')
    
    scene.add_wall([0, W/2, 1.5], [L, 0.3, 3.0], 'concrete')
    scene.add_wall([0, -W/2, 1.5], [L, 0.3, 3.0], 'concrete')
    
    return scene

def create_selleck_scene(dims=None):
    scene = BuildingSceneBuilder('selleck')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(388.06)
        W = ft_to_m(207.22)
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,4.0], [L, W, 0.1], 'drywall')
    
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
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L, W, 0.1], 'concrete')
    
    scene.add_room([10, 5, 0], [12, 10], 3.5, 'brick')
    
    return scene

def create_burnett_scene(dims=None):
    scene = BuildingSceneBuilder('burnett')
    if dims and dims.get('length_ft') and dims.get('width_ft'):
        L = ft_to_m(dims['length_ft'])
        W = ft_to_m(dims['width_ft'])
    else:
        L = ft_to_m(237.25)
        W = ft_to_m(77.01)
    
    scene.add_floor([0,0,0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0,0,3.5], [L, W, 0.1], 'concrete')
    
    scene.add_wall([0, 1.5, 1.75], [L, 0.15, 3.5], 'brick')
    scene.add_wall([0, -1.5, 1.75], [L, 0.15, 3.5], 'brick')
    
    return scene

def create_stadium_scene(dims=None):
    scene = BuildingSceneBuilder('memorial_stadium')
    L, W = 100.0, 20.0
    H = 5.0
    
    scene.add_floor([0,0,0], [L, W, 0.5], 'concrete')
    scene.add_ceiling([0,0,H], [L, W, 0.5], 'concrete')
    
    scene.add_wall([0, -W/2, H/2], [L, 1.0, H], 'concrete')
    
    return scene

def main():
    print("Generating scenes based on CSV data with accurate measurements...")
    
    building_dims = load_building_dimensions('Wireless Communications - Data Collection - Data-2.csv')
    
    building_map = {
        'Kiewit': (create_kiewit_scene, 'kiewit'),
        'Kauffman': (create_kauffman_scene, 'kauffman'),
        'Bessey': (create_bessey_scene, 'bessey'),
        'Love Library': (create_love_library_scene, 'love_library'),
        'Love Library South': (create_love_library_scene, 'love_library'),
        'Adele Coryell': (create_love_library_scene, 'love_library'),
        'Carolyn Pope Edwards': (create_edwards_scene, 'carolyn_pope_edwards'),
        'Avery': (create_avery_scene, 'avery'),
        'Coliseum': (create_coliseum_scene, 'coliseum'),
        'Union': (create_union_scene, 'union'),
        'Oldfather': (create_oldfather_scene, 'oldfather'),
        'Selleck': (create_selleck_scene, 'selleck'),
        'Brace': (create_brace_scene, 'brace'),
        'Burnett': (create_burnett_scene, 'burnett'),
        'Memorial Stadium': (create_stadium_scene, 'memorial_stadium')
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