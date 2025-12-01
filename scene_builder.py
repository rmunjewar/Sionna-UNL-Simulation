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

    def generate_xml(self, output_dir='scenes'):
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{self.building_name}.xml')
        xml_content = self._build_header() + self._build_materials() + "".join([self._obj_to_xml(o) for o in self.objects]) + self._build_footer()
        with open(filename, 'w') as f:
            f.write(xml_content)
        print(f"Generated: {filename}")
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
    """Convert feet to meters"""
    return ft * 0.3048

# Material type mapping for each building
# Exterior walls and structural elements use the building's material type
BUILDING_MATERIALS = {
    'Kiewit': 'glass_metal',  # Glass + Metal (windows and glass walls inside are glass)
    'Adele Coryell': 'brick',
    'Love Library South': 'concrete',
    'Selleck': 'brick',
    'Brace': 'brick',
    'Bessey': 'brick',
    'Union': 'brick',
    'Hamilton': 'concrete',
    'Oldfather': 'concrete',
    'Memorial Stadium': 'concrete',
    'Burnett': 'brick',
}

# Measurement locations with estimated coordinates based on sketches
# Format: [x_meters, y_meters, z_meters (height)]
MEASUREMENT_LOCATIONS = {
    'Kiewit': {
        'A310': [20.0, 40.0, 1.5],  # Northeast area, inside classroom
        'Lobby': [0.0, -45.0, 1.5],  # Near entrance, southern area
    },
    'Hamilton': {
        'Stairwell': [15.0, 20.0, 1.5],  # Near stairwell
        'Chemistry Resource Center': [18.0, 25.0, 1.5],  # Near stairwell
    },
    'Adele Coryell': {
        'Schmoker Learning Commons': [0.0, 5.0, 1.5],  # Center of open area
    },
    'Love Library South': {
        'Lobby': [0.0, -15.0, 1.5],  # Central lobby area
    },
    'Selleck': {
        'Dining Hall': [0.0, 0.0, 1.5],  # Center of large dining area
    },
    'Brace': {
        '210': [10.0, 25.0, 1.5],  # Inside classroom 210
    },
    'Bessey': {
        'Hallway': [0.0, 0.0, 1.5],  # Central hallway
    },
    'Union': {
        'Fireplace': [0.0, 0.0, 1.5],  # Fireplace area
    },
    'Oldfather': {
        '204': [10.0, 15.0, 1.5],  # Inside classroom 204
    },
    'Burnett': {
        'Hallway': [0.0, 0.0, 1.5],  # Central hallway
    },
    'Memorial Stadium': {
        'Concourse': [0.0, 0.0, 1.5],  # Concourse area
    }
}

def create_kiewit_scene():
    """
    Kiewit Hall - Engineering building
    Dimensions: 246.91 ft × 363.53 ft
    Measurements: A310 (classroom), Lobby
    Material: Glass + Metal (windows and glass walls inside building are glass)
    """
    scene = BuildingSceneBuilder('kiewit')
    L = ft_to_m(246.91)  # ~75.3m
    W = ft_to_m(363.53)  # ~110.8m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    # Exterior walls (metal frame with glass)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'metal')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'metal')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'metal')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'metal')
    
    # Large windows (glass walls)
    scene.add_window([0, W/2, 1.75], [L*0.9, 0.1, 2.5], 'glass')
    scene.add_window([0, -W/2, 1.75], [L*0.9, 0.1, 2.5], 'glass')
    scene.add_window([L/2, 0, 1.75], [0.1, W*0.9, 2.5], 'glass')
    scene.add_window([-L/2, 0, 1.75], [0.1, W*0.9, 2.5], 'glass')
    
    # Central hallway (runs north-south) - glass walls inside building
    hallway_width = 3.0
    scene.add_wall([10, 0, 1.75], [0.15, W*0.8, 3.5], 'glass')
    scene.add_wall([-10, 0, 1.75], [0.15, W*0.8, 3.5], 'glass')
    
    # Room A310 area (northeast, classroom with 4 walls)
    # Position: approximately at [20, 40]
    room_x, room_y = 20.0, 40.0
    room_width = 8.0
    room_length = 10.0
    
    # A310 walls (glass walls inside building)
    scene.add_wall([room_x, room_y + room_length/2, 1.75], [room_width, 0.15, 3.5], 'glass')
    scene.add_wall([room_x, room_y - room_length/2, 1.75], [room_width, 0.15, 3.5], 'glass')
    scene.add_wall([room_x + room_width/2, room_y, 1.75], [0.15, room_length, 3.5], 'glass')
    scene.add_wall([room_x - room_width/2, room_y, 1.75], [0.15, room_length, 3.5], 'glass')
    scene.add_door([room_x - room_width/2, room_y, 1.75], [0.15, 1.0, 2.5], 'wood')
    
    # Stairwell (concrete, blocks signal)
    scene.add_wall([-25, -40, 1.75], [4.0, 4.0, 3.5], 'concrete')
    
    # Lobby area (more open, just structural columns)
    scene.add_wall([0, -45, 1.75], [1.0, 1.0, 3.5], 'concrete')
    
    return scene

def create_adele_coryell_scene():
    """
    Adele Coryell Hall - Learning Commons
    Dimensions: 250.30 ft × 134.81 ft
    Measurements: Schmoker Learning Commons (open study area)
    Material: brick
    """
    scene = BuildingSceneBuilder('adele_coryell')
    L = ft_to_m(250.30)  # ~76.3m
    W = ft_to_m(134.81)  # ~41.1m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    # Exterior walls (brick)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    # Large windows on perimeter (typical for modern learning commons)
    scene.add_window([0, W/2, 1.75], [L*0.85, 0.1, 2.5], 'glass')
    scene.add_window([0, -W/2, 1.75], [L*0.85, 0.1, 2.5], 'glass')
    
    # Minimal interior walls (open study space)
    # Just a few partial walls for study nooks
    scene.add_wall([L/4, 0, 1.75], [0.15, W*0.3, 2.0], 'drywall')
    scene.add_wall([-L/4, 0, 1.75], [0.15, W*0.3, 2.0], 'drywall')
    
    # Structural columns
    scene.add_wall([15, 10, 1.75], [0.8, 0.8, 3.5], 'concrete')
    scene.add_wall([-15, 10, 1.75], [0.8, 0.8, 3.5], 'concrete')
    
    return scene

def create_love_library_south_scene():
    """
    Love Library South
    Dimensions: 213.44 ft × 141.25 ft
    Measurements: Lobby
    Material: concrete
    """
    scene = BuildingSceneBuilder('love_library_south')
    L = ft_to_m(213.44)  # ~65.1m
    W = ft_to_m(141.25)  # ~43.1m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'concrete')
    
    # Exterior walls (concrete)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'concrete')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'concrete')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'concrete')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'concrete')
    
    # Windows
    scene.add_window([0, W/2, 1.75], [L*0.9, 0.1, 2.5], 'glass')
    scene.add_window([0, -W/2, 1.75], [L*0.9, 0.1, 2.5], 'glass')
    
    # Support columns (library structure)
    col_spacing = 15.0
    for x in [-20, 0, 20]:
        for y in [-10, 10]:
            scene.add_wall([x, y, 1.75], [0.6, 0.6, 3.5], 'concrete')
    
    # Stairwell
    scene.add_wall([25, -15, 1.75], [4.0, 4.0, 3.5], 'concrete')
    
    return scene

def create_selleck_scene():
    """
    Selleck Dining Hall
    Dimensions: 207.22 ft × 388.06 ft
    Measurements: Dining Hall (large open area)
    Material: brick
    """
    scene = BuildingSceneBuilder('selleck')
    L = ft_to_m(207.22)  # ~63.2m
    W = ft_to_m(388.06)  # ~118.3m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 4.0], [L, W, 0.1], 'drywall')  # Higher ceiling for dining hall
    
    # Exterior walls (brick)
    wall_thick = 0.4
    scene.add_wall([0, W/2, 2.0], [L, wall_thick, 4.0], 'brick')
    scene.add_wall([0, -W/2, 2.0], [L, wall_thick, 4.0], 'brick')
    scene.add_wall([L/2, 0, 2.0], [wall_thick, W, 4.0], 'brick')
    scene.add_wall([-L/2, 0, 2.0], [wall_thick, W, 4.0], 'brick')
    
    # Very open interior (typical dining hall)
    # Just structural columns
    col_spacing = 20.0
    for x in [-20, 0, 20]:
        for y in [-40, -20, 0, 20, 40]:
            scene.add_wall([x, y, 2.0], [1.0, 1.0, 4.0], 'concrete')
    
    # Kitchen/serving area (more walls)
    scene.add_wall([0, -50, 2.0], [L*0.6, 0.15, 4.0], 'drywall')
    
    return scene

def create_brace_scene():
    """
    Brace Hall - Laboratory building
    Dimensions: 123.03 ft × 220.32 ft
    Measurements: Room 210
    Material: brick
    """
    scene = BuildingSceneBuilder('brace')
    L = ft_to_m(123.03)  # ~37.5m
    W = ft_to_m(220.32)  # ~67.2m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'concrete')
    
    # Exterior walls (brick)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    # Central hallway
    scene.add_wall([0, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    
    # Room 210 (classroom/lab)
    room_x, room_y = 10.0, 25.0
    room_width = 8.0
    room_length = 10.0
    
    scene.add_wall([room_x, room_y + room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
    scene.add_wall([room_x, room_y - room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
    scene.add_wall([room_x + room_width/2, room_y, 1.75], [0.15, room_length, 3.5], 'drywall')
    scene.add_wall([room_x - room_width/2, room_y, 1.75], [0.15, room_length, 3.5], 'drywall')
    scene.add_door([room_x - room_width/2, room_y, 1.75], [0.15, 1.0, 2.5], 'wood')
    
    return scene

def create_hamilton_scene():
    """
    Hamilton Hall
    Dimensions: 160 ft × 485.78 ft (doubled from original to match CSV area ~73,600 sq ft)
    Measurements: Stairwell, Chemistry Resource Center
    Material: concrete
    """
    scene = BuildingSceneBuilder('hamilton')
    L = ft_to_m(160)  # ~48.8m (doubled from 80ft)
    W = ft_to_m(485.78)  # ~148.2m (doubled from 242.89ft)
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'concrete')
    
    # Exterior walls (concrete)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'concrete')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'concrete')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'concrete')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'concrete')
    
    # Stairwell area (hollow room with 4 thin concrete walls)
    # Receiver is at (15, 20), so we create a hollow 4x5m room centered there
    stair_x, stair_y = 15.0, 20.0
    stair_width = 4.0  # meters
    stair_length = 5.0  # meters
    wall_thickness = 0.2  # meters (thin walls)
    
    # North wall (top)
    scene.add_wall([stair_x, stair_y + stair_length/2, 1.75], [stair_width, wall_thickness, 3.5], 'concrete')
    # South wall (bottom)
    scene.add_wall([stair_x, stair_y - stair_length/2, 1.75], [stair_width, wall_thickness, 3.5], 'concrete')
    # East wall (right)
    scene.add_wall([stair_x + stair_width/2, stair_y, 1.75], [wall_thickness, stair_length, 3.5], 'concrete')
    # West wall (left)
    scene.add_wall([stair_x - stair_width/2, stair_y, 1.75], [wall_thickness, stair_length, 3.5], 'concrete')
    
    # Chemistry Resource Center (5 feet from stairwell)
    crc_x, crc_y = 18.0, 25.0
    scene.add_wall([crc_x, crc_y + 4, 1.75], [8.0, 0.15, 3.5], 'drywall')
    scene.add_wall([crc_x, crc_y - 4, 1.75], [8.0, 0.15, 3.5], 'drywall')
    
    # Central hallway
    scene.add_wall([0, 0, 1.75], [L*0.7, 0.15, 3.5], 'drywall')
    
    return scene

def create_bessey_scene():
    """
    Bessey Hall
    Dimensions: 239.09 ft × 319.46 ft
    Measurements: Hallway
    Material: brick
    """
    scene = BuildingSceneBuilder('bessey')
    L = ft_to_m(239.09)  # ~72.9m
    W = ft_to_m(319.46)  # ~97.4m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    # Exterior walls (brick)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    # Central hallway
    scene.add_wall([0, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    scene.add_wall([10, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    scene.add_wall([-10, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    
    # Structural columns
    for x in [-15, 0, 15]:
        for y in [-30, 0, 30]:
            scene.add_wall([x, y, 1.75], [0.8, 0.8, 3.5], 'concrete')
    
    return scene

def create_union_scene():
    """
    Union Building
    Dimensions: 256.88 ft × 313.54 ft
    Measurements: Fireplace
    Material: brick
    """
    scene = BuildingSceneBuilder('union')
    L = ft_to_m(256.88)  # ~78.3m
    W = ft_to_m(313.54)  # ~95.5m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    # Exterior walls (brick)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    # Open area with fireplace (central)
    # Fireplace area
    scene.add_wall([0, 0, 1.75], [3.0, 2.0, 2.5], 'brick')
    
    # Support columns
    for x in [-20, 0, 20]:
        for y in [-25, 0, 25]:
            scene.add_wall([x, y, 1.75], [0.8, 0.8, 3.5], 'concrete')
    
    return scene

def create_oldfather_scene():
    """
    Oldfather Hall
    Dimensions: 74.88 ft × 126.55 ft
    Measurements: Room 204
    Material: concrete
    """
    scene = BuildingSceneBuilder('oldfather')
    L = ft_to_m(74.88)  # ~22.8m
    W = ft_to_m(126.55)  # ~38.6m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'concrete')
    
    # Exterior walls (concrete)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'concrete')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'concrete')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'concrete')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'concrete')
    
    # Central hallway
    scene.add_wall([0, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    
    # Room 204 (classroom)
    room_x, room_y = 10.0, 15.0
    room_width = 7.0
    room_length = 9.0
    
    scene.add_wall([room_x, room_y + room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
    scene.add_wall([room_x, room_y - room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
    scene.add_wall([room_x + room_width/2, room_y, 1.75], [0.15, room_length, 3.5], 'drywall')
    scene.add_wall([room_x - room_width/2, room_y, 1.75], [0.15, room_length, 3.5], 'drywall')
    scene.add_door([room_x - room_width/2, room_y, 1.75], [0.15, 1.0, 2.5], 'wood')
    
    return scene

def create_burnett_scene():
    """
    Burnett Hall
    Dimensions: 77.01 ft × 237.25 ft
    Measurements: Hallway
    Material: brick
    """
    scene = BuildingSceneBuilder('burnett')
    L = ft_to_m(77.01)  # ~23.5m
    W = ft_to_m(237.25)  # ~72.3m
    
    # Floor and ceiling
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    # Exterior walls (brick)
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    # Central hallway
    scene.add_wall([0, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    
    # Classroom areas along hallway
    for y_offset in [-30, -15, 0, 15, 30]:
        room_width = 6.0
        room_length = 8.0
        scene.add_wall([8, y_offset + room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
        scene.add_wall([8, y_offset - room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
        scene.add_wall([8 + room_width/2, y_offset, 1.75], [0.15, room_length, 3.5], 'drywall')
    
    return scene

def create_memorial_stadium_scene():
    """
    Memorial Stadium
    Dimensions: 498.18 ft × 732.38 ft
    Measurements: Concourse
    Material: concrete
    """
    scene = BuildingSceneBuilder('memorial_stadium')
    L = ft_to_m(498.18)  # ~151.8m
    W = ft_to_m(732.38)  # ~223.2m
    
    # Floor and ceiling (stadium concourse)
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 5.0], [L, W, 0.1], 'concrete')  # Higher ceiling for stadium
    
    # Exterior walls (concrete)
    wall_thick = 0.4
    scene.add_wall([0, W/2, 2.5], [L, wall_thick, 5.0], 'concrete')
    scene.add_wall([0, -W/2, 2.5], [L, wall_thick, 5.0], 'concrete')
    scene.add_wall([L/2, 0, 2.5], [wall_thick, W, 5.0], 'concrete')
    scene.add_wall([-L/2, 0, 2.5], [wall_thick, W, 5.0], 'concrete')
    
    # Large open concourse area with support columns
    col_spacing = 30.0
    for x in range(-60, 61, 30):
        for y in range(-90, 91, 30):
            scene.add_wall([x, y, 2.5], [1.5, 1.5, 5.0], 'concrete')
    
    # Entry areas (more open)
    scene.add_window([0, W/2, 2.5], [L*0.3, 0.1, 3.0], 'glass')
    scene.add_window([0, -W/2, 2.5], [L*0.3, 0.1, 3.0], 'glass')
    
    return scene

def main():
    print("="*70)
    print("GENERATING REALISTIC BUILDING SCENES")
    print("Based on actual measurements and floor plan sketches but using approximations")
    print("="*70)
    
    buildings = [
        ('Kiewit', create_kiewit_scene),
        ('Hamilton', create_hamilton_scene),
        ('Bessey', create_bessey_scene),
        ('Adele Coryell', create_adele_coryell_scene),
        ('Love Library South', create_love_library_south_scene),
        ('Union', create_union_scene),
        ('Oldfather', create_oldfather_scene),
        ('Selleck', create_selleck_scene),
        ('Brace', create_brace_scene),
        ('Burnett', create_burnett_scene),
        ('Memorial Stadium', create_memorial_stadium_scene),
    ]
    
    for building_name, builder_func in buildings:
        try:
            print(f"\nGenerating {building_name}...")
            scene = builder_func()
            scene.generate_xml()
        except Exception as e:
            print(f"Error generating {building_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SCENE GENERATION COMPLETE")
    print("="*70)
    
    # Print measurement locations for reference
    print("\nMeasurement Locations (in building coordinates):")
    for building, locations in MEASUREMENT_LOCATIONS.items():
        print(f"\n{building}:")
        for location, coords in locations.items():
            print(f"  {location}: x={coords[0]:.1f}m, y={coords[1]:.1f}m, z={coords[2]:.1f}m")

if __name__ == "__main__":
    main()