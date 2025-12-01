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

BUILDING_MATERIALS = {
    'Kiewit': 'glass_metal',
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

MEASUREMENT_LOCATIONS = {
    'Kiewit': {
        'A310': [20.0, 40.0, 1.5],
        'Lobby': [0.0, -45.0, 1.5],
    },
    'Hamilton': {
        'Stairwell': [15.0, 20.0, 1.5],
        'Chemistry Resource Center': [18.0, 25.0, 1.5],
    },
    'Adele Coryell': {
        'Schmoker Learning Commons': [0.0, 5.0, 1.5],
    },
    'Love Library South': {
        'Lobby': [0.0, -15.0, 1.5],
    },
    'Selleck': {
        'Dining Hall': [0.0, 0.0, 1.5],
    },
    'Brace': {
        '210': [10.0, 25.0, 1.5],
    },
    'Bessey': {
        'Hallway': [0.0, 0.0, 1.5],
    },
    'Union': {
        'Fireplace': [0.0, 0.0, 1.5],
    },
    'Oldfather': {
        '204': [10.0, 15.0, 1.5],
    },
    'Burnett': {
        'Hallway': [0.0, 0.0, 1.5],
    },
    'Memorial Stadium': {
        'Concourse': [0.0, 0.0, 1.5],
    }
}

def create_kiewit_scene():
    scene = BuildingSceneBuilder('kiewit_updated')
    
    Width_X = ft_to_m(363.53)
    Depth_Y = ft_to_m(246.91)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [Width_X, Depth_Y, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_window([0, Depth_Y/2, 1.75], [Width_X, 0.1, 3.5], 'glass')
    scene.add_window([0, -Depth_Y/2, 1.75], [Width_X, 0.1, 3.5], 'glass')
    scene.add_window([Width_X/2, 0, 1.75], [0.1, Depth_Y, 3.5], 'glass')
    scene.add_window([-Width_X/2, 0, 1.75], [0.1, Depth_Y, 3.5], 'glass')

    core_w = 20.0
    core_d = 12.0
    core_x = -5.0
    core_y = 15.0
    
    scene.add_wall([core_x, core_y - core_d/2, 1.75], [core_w, 0.3, 3.5], 'concrete')
    scene.add_wall([core_x, core_y + core_d/2, 1.75], [core_w, 0.3, 3.5], 'concrete')
    scene.add_wall([core_x + core_w/2, core_y, 1.75], [0.3, core_d, 3.5], 'concrete')
    scene.add_wall([core_x - core_w/2, core_y, 1.75], [0.3, core_d, 3.5], 'concrete')

    room_w = 22.0
    room_d = 14.0
    
    room_center_x = 35.0 
    room_center_y = 5.0 
    
    scene.add_wall([room_center_x - room_w/2, room_center_y, 1.75], [0.1, room_d, 3.5], 'drywall')
    
    scene.add_wall([room_center_x, room_center_y + room_d/2, 1.75], [room_w, 0.1, 3.5], 'drywall')
    
    scene.add_wall([room_center_x, room_center_y - room_d/2, 1.75], [room_w, 0.1, 3.5], 'drywall')

    scan_x = room_center_x - (room_w/2) + 3.0
    scan_y = room_center_y + 2.0
    scan_z = 1.5
    
    print(f"\n[Kiewit Configuration]")
    print(f"Room A.310 Center: ({room_center_x}, {room_center_y})")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    print(f"Suggested Transmitter Location (Lobby/Elevator): x={core_x}, y={core_y - 15.0}, z=1.5")

    return scene

def create_adele_coryell_scene():
    scene = BuildingSceneBuilder('adele_coryell_updated')
    
    Width_X = ft_to_m(250.30)
    Depth_Y = ft_to_m(134.81)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [Width_X, Depth_Y, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_wall([0, Depth_Y/2, 1.75], [Width_X, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -Depth_Y/2, 1.75], [Width_X, wall_thick, 3.5], 'brick')
    scene.add_wall([Width_X/2, 0, 1.75], [wall_thick, Depth_Y, 3.5], 'brick')
    scene.add_wall([-Width_X/2, 0, 1.75], [wall_thick, Depth_Y, 3.5], 'brick')

    scene.add_window([0, Depth_Y/2, 1.75], [6.0, 0.1, 2.5], 'glass')
    scene.add_window([-5, -Depth_Y/2, 1.75], [6.0, 0.1, 2.5], 'glass')

    rr_w = 12.0
    rr_d = 8.0
    rr_x = -10.0
    rr_y = -10.0
    
    scene.add_wall([rr_x, rr_y + rr_d/2, 1.75], [rr_w, 0.15, 3.5], 'concrete')
    scene.add_wall([rr_x, rr_y - rr_d/2, 1.75], [rr_w, 0.15, 3.5], 'concrete')
    scene.add_wall([rr_x + rr_w/2, rr_y, 1.75], [0.15, rr_d, 3.5], 'concrete')
    scene.add_wall([rr_x - rr_w/2, rr_y, 1.75], [0.15, rr_d, 3.5], 'concrete')
    
    scene.add_wall([-25.0, -5.0, 1.75], [0.15, 20.0, 3.5], 'glass')

    scene.add_wall([-15.0, 10.0, 1.75], [30.0, 0.15, 3.5], 'drywall')
    
    dlc_x_center = 20.0
    dlc_y_center = -5.0
    dlc_width = 30.0
    dlc_depth = 25.0
    
    scene.add_wall([5.0, -5.0, 1.75], [0.15, 25.0, 3.5], 'drywall')
    
    scene.add_wall([20.0, 8.0, 1.75], [30.0, 0.15, 3.5], 'drywall')

    scan_x = 22.0
    scan_y = -12.0
    scan_z = 1.5
    
    print(f"\n[Adele Coryell Configuration]")
    print(f"Digital Learning Center (DLC) Area: East Side")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    
    return scene

def create_love_library_south_scene():
    scene = BuildingSceneBuilder('love_library_south_updated')
    
    Width_X = ft_to_m(213.44)
    Depth_Y = ft_to_m(141.25)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 4.0], [Width_X, Depth_Y, 0.1], 'concrete')
    
    wall_thick = 0.3
    scene.add_wall([0, Depth_Y/2, 2.0], [Width_X, wall_thick, 4.0], 'concrete')
    scene.add_wall([0, -Depth_Y/2, 2.0], [Width_X, wall_thick, 4.0], 'concrete')
    scene.add_wall([Width_X/2, 0, 2.0], [wall_thick, Depth_Y, 4.0], 'concrete')
    scene.add_wall([-Width_X/2, 0, 2.0], [wall_thick, Depth_Y, 4.0], 'concrete')

    scene.add_window([0, Depth_Y/2, 2.0], [8.0, 0.1, 3.0], 'glass')
    scene.add_window([0, -Depth_Y/2, 2.0], [8.0, 0.1, 3.0], 'glass')

    core_w = 20.0
    core_d = 8.0
    core_x = 0.0
    core_y = -5.0
    
    scene.add_wall([core_x, core_y + core_d/2, 2.0], [core_w, 0.2, 4.0], 'concrete')
    scene.add_wall([core_x, core_y - core_d/2, 2.0], [core_w, 0.2, 4.0], 'concrete')
    scene.add_wall([core_x + core_w/2, core_y, 2.0], [0.2, core_d, 4.0], 'concrete')
    scene.add_wall([core_x - core_w/2, core_y, 2.0], [0.2, core_d, 4.0], 'concrete')
    
    aud_w = 25.0
    aud_d = 15.0
    aud_x = -15.0
    aud_y = 10.0
    
    scene.add_wall([aud_x + aud_w/2, aud_y, 2.0], [0.2, aud_d, 4.0], 'concrete')
    scene.add_wall([aud_x, aud_y - aud_d/2, 2.0], [aud_w, 0.2, 4.0], 'concrete')
    
    ask_w = 10.0
    ask_d = 12.0
    ask_x = 15.0
    ask_y = 10.0
    
    scene.add_wall([ask_x - ask_w/2, ask_y, 2.0], [0.2, ask_d, 4.0], 'wood')
    
    scene.add_wall([0.0, -25.0, 2.0], [30.0, 0.2, 4.0], 'drywall')

    scan_x = 0.0
    scan_y = 5.0
    scan_z = 1.5
    
    print(f"\n[Love Library South Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    print(f"  (Located in Central Atrium, North of Elevators)")
    
    return scene

def create_selleck_scene():
    scene = BuildingSceneBuilder('selleck_updated')
    
    Width_X = ft_to_m(207.22)
    Depth_Y = ft_to_m(388.06)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 4.0], [Width_X, Depth_Y, 0.1], 'drywall')
    
    wall_thick = 0.4
    scene.add_wall([0, Depth_Y/2, 2.0], [Width_X, wall_thick, 4.0], 'brick')
    scene.add_wall([0, -Depth_Y/2, 2.0], [Width_X, wall_thick, 4.0], 'brick')
    scene.add_wall([Width_X/2, 0, 2.0], [wall_thick, Depth_Y, 4.0], 'brick')
    scene.add_wall([-Width_X/2, 0, 2.0], [wall_thick, Depth_Y, 4.0], 'brick')
    
    col_spacing_x = 10.0
    col_spacing_y = 12.0
    
    for x in np.arange(-20, 21, col_spacing_x):
        for y in np.arange(-40, 41, col_spacing_y):
            if abs(x - (-10)) < 3 and abs(y - 0) < 3:
                continue
            scene.add_wall([x, y, 2.0], [0.6, 0.6, 4.0], 'concrete')
            
    scene.add_wall([15.0, 0, 2.0], [0.2, 60.0, 4.0], 'metal')
    scene.add_wall([25.0, 0, 2.0], [10.0, 40.0, 4.0], 'concrete')

    scan_x = -12.0
    scan_y = 0.0
    scan_z = 1.5
    
    print(f"\n[Selleck Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    
    return scene

def create_brace_scene():
    scene = BuildingSceneBuilder('brace_updated')
    
    Width_X = ft_to_m(123.03)
    Depth_Y = ft_to_m(220.32)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [Width_X, Depth_Y, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_wall([0, Depth_Y/2, 1.75], [Width_X, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -Depth_Y/2, 1.75], [Width_X, wall_thick, 3.5], 'brick')
    scene.add_wall([Width_X/2, 0, 1.75], [wall_thick, Depth_Y, 3.5], 'brick')
    scene.add_wall([-Width_X/2, 0, 1.75], [wall_thick, Depth_Y, 3.5], 'brick')
    
    corridor_y = 5.0 
    scene.add_wall([-10.0, corridor_y, 1.75], [Width_X/1.5, 0.2, 3.5], 'brick')
    
    scene.add_wall([5.0, -10.0, 1.75], [0.2, 40.0, 3.5], 'brick')
    
    room_x_center = -10.0
    room_y_center = -15.0
    room_w = 12.0
    room_d = 10.0
    
    scene.add_wall([room_x_center, room_y_center + room_d/2, 1.75], [room_w, 0.15, 3.5], 'drywall')
    scene.add_wall([room_x_center + room_w/2, room_y_center, 1.75], [0.15, room_d, 3.5], 'drywall')
    
    scan_x = -10.0
    scan_y = -18.0
    scan_z = 1.5
    
    print(f"\n[Brace Hall Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    
    return scene

def create_oldfather_scene():
    scene = BuildingSceneBuilder('oldfather_updated')
    
    Width_X = ft_to_m(126.45)
    Depth_Y = ft_to_m(67.05)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [Width_X, Depth_Y, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_wall([0, Depth_Y/2, 1.75], [Width_X, wall_thick, 3.5], 'concrete')
    scene.add_wall([0, -Depth_Y/2, 1.75], [Width_X, wall_thick, 3.5], 'concrete')
    scene.add_wall([Width_X/2, 0, 1.75], [wall_thick, Depth_Y, 3.5], 'concrete')
    scene.add_wall([-Width_X/2, 0, 1.75], [wall_thick, Depth_Y, 3.5], 'concrete')
    
    scene.add_window([0, Depth_Y/2, 1.75], [Width_X*0.8, 0.1, 2.0], 'glass')
    
    core_w = 12.0
    core_d = 8.0
    
    scene.add_wall([0, 0, 1.75], [core_w, core_d, 3.5], 'concrete')
    
    room_x = -12.0
    room_y = 5.0
    room_w = 8.0
    room_d = 6.0
    
    scene.add_wall([room_x + room_w/2, room_y, 1.75], [0.15, room_d, 3.5], 'drywall')
    scene.add_wall([room_x, room_y - room_d/2, 1.75], [room_w, 0.15, 3.5], 'drywall')
    
    scan_x = -14.0
    scan_y = 6.0
    scan_z = 1.5
    
    print(f"\n[Oldfather Hall Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    
    return scene

def create_hamilton_scene():
    scene = BuildingSceneBuilder('hamilton_updated')
    
    wing_ns_w = 25.0
    wing_ns_h = 60.0
    
    wing_ew_w = 50.0
    wing_ew_h = 25.0
    
    scene.add_floor([0, 0, 0], [80.0, 80.0, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [80.0, 80.0, 0.1], 'drywall')
    
    core_x = 10.0
    core_y = 0.0
    scene.add_wall([core_x, core_y, 1.75], [10.0, 8.0, 3.5], 'concrete')
    
    scene.add_wall([-20.0, 2.0, 1.75], [50.0, 0.15, 3.5], 'drywall')
    scene.add_wall([-20.0, -2.0, 1.75], [50.0, 0.15, 3.5], 'drywall')
    
    room_x = -25.0
    room_y = -8.0
    
    scene.add_wall([room_x, room_y + 4.0, 1.75], [10.0, 0.15, 3.5], 'drywall')
    scene.add_wall([room_x + 5.0, room_y, 1.75], [0.15, 8.0, 3.5], 'drywall')
    
    scan_x = -25.0
    scan_y = -8.0
    scan_z = 1.5
    
    print(f"\n[Hamilton Hall Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    
    return scene

def create_bessey_scene():
    scene = BuildingSceneBuilder('bessey')
    L = ft_to_m(239.09)
    W = ft_to_m(319.46)
    
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    scene.add_wall([0, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    scene.add_wall([10, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    scene.add_wall([-10, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    
    for x in [-15, 0, 15]:
        for y in [-30, 0, 30]:
            scene.add_wall([x, y, 1.75], [0.8, 0.8, 3.5], 'concrete')
    
    return scene

def create_union_scene():
    scene = BuildingSceneBuilder('union_updated')
    
    Width_X = ft_to_m(256.88)
    Depth_Y = ft_to_m(313.54)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 4.0], [Width_X, Depth_Y, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_wall([0, Depth_Y/2, 2.0], [Width_X, wall_thick, 4.0], 'brick')
    scene.add_wall([0, -Depth_Y/2, 2.0], [Width_X, wall_thick, 4.0], 'brick')
    scene.add_wall([Width_X/2, 0, 2.0], [wall_thick, Depth_Y, 4.0], 'brick')
    scene.add_wall([-Width_X/2, 0, 2.0], [wall_thick, Depth_Y, 4.0], 'brick')
    
    crib_x = -5.0
    crib_y = -30.0
    crib_w = 15.0
    crib_d = 12.0
    
    scene.add_wall([crib_x, crib_y + crib_d/2, 2.0], [crib_w, 0.15, 4.0], 'drywall')
    scene.add_wall([crib_x + crib_w/2, crib_y, 2.0], [0.15, crib_d, 4.0], 'drywall')
    
    scene.add_wall([0.0, 10.0, 2.0], [8.0, 8.0, 4.0], 'concrete')
    
    scan_x = -8.0
    scan_y = -30.0
    scan_z = 1.5
    
    print(f"\n[Union Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    
    return scene

def create_burnett_scene():
    scene = BuildingSceneBuilder('burnett')
    L = ft_to_m(77.01)
    W = ft_to_m(237.25)
    
    scene.add_floor([0, 0, 0], [L, W, 0.3], 'concrete')
    scene.add_ceiling([0, 0, 3.5], [L, W, 0.1], 'drywall')
    
    wall_thick = 0.3
    scene.add_wall([0, W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([0, -W/2, 1.75], [L, wall_thick, 3.5], 'brick')
    scene.add_wall([L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    scene.add_wall([-L/2, 0, 1.75], [wall_thick, W, 3.5], 'brick')
    
    scene.add_wall([0, 0, 1.75], [0.15, W*0.8, 3.5], 'drywall')
    
    for y_offset in [-30, -15, 0, 15, 30]:
        room_width = 6.0
        room_length = 8.0
        scene.add_wall([8, y_offset + room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
        scene.add_wall([8, y_offset - room_length/2, 1.75], [room_width, 0.15, 3.5], 'drywall')
        scene.add_wall([8 + room_width/2, y_offset, 1.75], [0.15, room_length, 3.5], 'drywall')
    
    return scene

def create_memorial_stadium_scene():
    scene = BuildingSceneBuilder('memorial_stadium_updated')
    
    Width_X = ft_to_m(498.18)
    Depth_Y = ft_to_m(732.38)
    
    scene.add_floor([0, 0, 0], [Width_X, Depth_Y, 0.3], 'concrete')
    
    west_stand_x = -50.0 
    west_stand_width = 40.0
    west_stand_length = 150.0
    
    scene.add_wall([west_stand_x - 10, 0, 10.0], [5.0, west_stand_length, 20.0], 'concrete')
    
    club_z = 15.0
    club_width = 15.0
    club_length = 80.0
    
    scene.add_floor([west_stand_x, 0, club_z - 0.1], [club_width, club_length, 0.2], 'concrete')
    scene.add_ceiling([west_stand_x, 0, club_z + 4.0], [club_width, club_length, 0.1], 'drywall')
    
    scene.add_window([west_stand_x + club_width/2, 0, club_z + 2.0], [0.1, club_length, 4.0], 'glass')
    
    scene.add_wall([west_stand_x - club_width/2, 0, club_z + 2.0], [0.3, club_length, 4.0], 'concrete')

    scan_x = west_stand_x + 2.0
    scan_y = 20.0
    scan_z = club_z + 1.5
    
    print(f"\n[Memorial Stadium Configuration]")
    print(f"RED STAR LOCATION (Receiver): x={scan_x:.2f}, y={scan_y:.2f}, z={scan_z}")
    print(f"  (Located in West Stadium Indoor Club, Elevation ~15m)")
    
    return scene

def main():
    print("="*70)
    print("GENERATING REALISTIC BUILDING SCENES")
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