bl_info = {
    "name": "Particle Morph Addon",
    "author": "Ed Lally",
    "version": (1, 1, 0),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > Mesh Morph",
    "description": "Morphs one mesh into another",
    "category": "Object",
}

import bpy
import bmesh
import mathutils
import mathutils.noise
import random
import math
from bpy.app.handlers import persistent
from bpy.types import Scene


MORPH_DATA = {}

def generate_particle_distortion_offset(particle_index):
    """random direction for each particle"""
    import math
    
    # pseudo-random direction for this particle
    seed = particle_index * 12345.0
    
    # stable random direction vector
    offset_x = math.sin(seed * 0.1) * math.cos(seed * 0.07)
    offset_y = math.sin(seed * 0.13) * math.cos(seed * 0.11)  
    offset_z = math.sin(seed * 0.17) * math.cos(seed * 0.19)
    
    return (offset_x, offset_y, offset_z)

def apply_distortion_to_path(t, distortion_offset, distortion_strength):
    """add distortion to particle path"""
    if distortion_strength <= 0:
        return (0.0, 0.0, 0.0)
    
    # strongest at middle, zero at ends
    falloff = 4.0 * t * (1.0 - t)  # peaks at 0.5
    
    if falloff <= 0:
        return (0.0, 0.0, 0.0)
    
    # apply falloff and strength
    final_strength = falloff * distortion_strength * 0.2
    
    return (
        distortion_offset[0] * final_strength,
        distortion_offset[1] * final_strength,
        distortion_offset[2] * final_strength
    )

def calculate_guaranteed_safe_offsets(all_src_positions, all_tgt_positions, avoidance_distance=1.0):
    """avoid particle collisions"""
    num_particles = len(all_src_positions)
    safe_offsets = [(0.0, 0.0, 0.0)] * num_particles
    
    if num_particles <= 1:
        return safe_offsets
    
    # sample points along each path to check collisions
    path_samples = 20  # points to sample along each path
    
    for iteration in range(10):  # max adjustment iterations
        collision_found = False
        
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                # check if paths i and j might collide
                min_distance_found = float('inf')
                
                # sample points along both paths
                for sample in range(path_samples + 1):
                    t = sample / path_samples
                    
                    # position of particle i at time t
                    src_i = all_src_positions[i]
                    tgt_i = all_tgt_positions[i]
                    offset_i = safe_offsets[i]
                    
                    # parabolic falloff for collision avoidance
                    avoidance_falloff = 4.0 * t * (1.0 - t)
                    
                    pos_i_x = src_i[0] * (1 - t) + tgt_i[0] * t + offset_i[0] * avoidance_falloff
                    pos_i_y = src_i[1] * (1 - t) + tgt_i[1] * t + offset_i[1] * avoidance_falloff
                    pos_i_z = src_i[2] * (1 - t) + tgt_i[2] * t + offset_i[2] * avoidance_falloff
                    
                    # position of particle j at time t
                    src_j = all_src_positions[j]
                    tgt_j = all_tgt_positions[j]
                    offset_j = safe_offsets[j]
                    
                    pos_j_x = src_j[0] * (1 - t) + tgt_j[0] * t + offset_j[0] * avoidance_falloff
                    pos_j_y = src_j[1] * (1 - t) + tgt_j[1] * t + offset_j[1] * avoidance_falloff
                    pos_j_z = src_j[2] * (1 - t) + tgt_j[2] * t + offset_j[2] * avoidance_falloff
                    
                    # distance between particles at this time
                    dx = pos_i_x - pos_j_x
                    dy = pos_i_y - pos_j_y
                    dz = pos_i_z - pos_j_z
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    min_distance_found = min(min_distance_found, distance)
                
                # if too close, adjust offsets
                if min_distance_found < avoidance_distance:
                    collision_found = True
                    
                    # separation vector between average positions
                    avg_src_i = all_src_positions[i]
                    avg_tgt_i = all_tgt_positions[i]
                    avg_pos_i_x = (avg_src_i[0] + avg_tgt_i[0]) * 0.5
                    avg_pos_i_y = (avg_src_i[1] + avg_tgt_i[1]) * 0.5
                    avg_pos_i_z = (avg_src_i[2] + avg_tgt_i[2]) * 0.5
                    
                    avg_src_j = all_src_positions[j]
                    avg_tgt_j = all_tgt_positions[j]
                    avg_pos_j_x = (avg_src_j[0] + avg_tgt_j[0]) * 0.5
                    avg_pos_j_y = (avg_src_j[1] + avg_tgt_j[1]) * 0.5
                    avg_pos_j_z = (avg_src_j[2] + avg_tgt_j[2]) * 0.5
                    
                    # separation vector
                    sep_x = avg_pos_i_x - avg_pos_j_x
                    sep_y = avg_pos_i_y - avg_pos_j_y
                    sep_z = avg_pos_i_z - avg_pos_j_z
                    
                    sep_length = math.sqrt(sep_x*sep_x + sep_y*sep_y + sep_z*sep_z)
                    
                    if sep_length > 0:
                        # normalize separation vector
                        sep_x /= sep_length
                        sep_y /= sep_length
                        sep_z /= sep_length
                        
                        # required adjustment
                        required_separation = avoidance_distance - min_distance_found + 0.1  # safety margin
                        adjustment_magnitude = required_separation * 0.5  # split between both particles
                        
                        # adjust offsets to increase separation
                        adjustment_i_x = sep_x * adjustment_magnitude
                        adjustment_i_y = sep_y * adjustment_magnitude
                        adjustment_i_z = sep_z * adjustment_magnitude
                        
                        adjustment_j_x = -sep_x * adjustment_magnitude
                        adjustment_j_y = -sep_y * adjustment_magnitude
                        adjustment_j_z = -sep_z * adjustment_magnitude
                        
                        # apply adjustments
                        safe_offsets[i] = (
                            safe_offsets[i][0] + adjustment_i_x,
                            safe_offsets[i][1] + adjustment_i_y,
                            safe_offsets[i][2] + adjustment_i_z
                        )
                        
                        safe_offsets[j] = (
                            safe_offsets[j][0] + adjustment_j_x,
                            safe_offsets[j][1] + adjustment_j_y,
                            safe_offsets[j][2] + adjustment_j_z
                        )
        
        # if no collisions found, we're done
        if not collision_found:
            break
    
    return safe_offsets

def random_point_on_mesh(mesh_obj):
    """random point on mesh surface"""
    mesh = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    
    if not bm.faces:
        bm.free()
        return mathutils.Vector((0, 0, 0))
    
    # face areas for weighted selection
    face_areas = [face.calc_area() for face in bm.faces]
    total_area = sum(face_areas)
    
    if total_area == 0:
        bm.free()
        return mathutils.Vector((0, 0, 0))
    
    # select random face weighted by area
    rand_val = random.uniform(0, total_area)
    cumulative = 0
    selected_face = bm.faces[0]
    
    for i, area in enumerate(face_areas):
        cumulative += area
        if rand_val <= cumulative:
            selected_face = bm.faces[i]
            break
    
    # random point within the selected face
    verts = selected_face.verts
    if len(verts) >= 3:
        # barycentric coordinates for triangular face
        r1, r2 = random.random(), random.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        r3 = 1 - r1 - r2
        
        point = (r1 * verts[0].co + r2 * verts[1].co + r3 * verts[2].co)
    else:
        point = verts[0].co
    
    bm.free()
    return mesh_obj.matrix_world @ point

def generate_evenly_distributed_points_on_mesh(mesh_obj, count, min_distance=0.1):
    """evenly distributed points on mesh surface"""
    points = []
    max_attempts = min(30, count * 2)  # limit attempts to prevent lag
    
    # create bmesh once and reuse it
    mesh = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    
    if not bm.faces:
        bm.free()
        return []
    
    # pre-calculate face areas once
    face_areas = [face.calc_area() for face in bm.faces]
    total_area = sum(face_areas)
    
    if total_area == 0:
        bm.free()
        return []
    
    # build cumulative area array for fast face selection
    cumulative_areas = []
    cumulative = 0
    for area in face_areas:
        cumulative += area
        cumulative_areas.append(cumulative)
    
    matrix_world = mesh_obj.matrix_world
    
    for i in range(count):
        best_candidate = None
        best_distance = 0
        
        # try multiple candidates and pick the one furthest from existing points
        for attempt in range(max_attempts):
            # fast face selection using pre-calculated cumulative areas
            rand_val = random.uniform(0, total_area)
            selected_face_idx = 0
            for j, cum_area in enumerate(cumulative_areas):
                if rand_val <= cum_area:
                    selected_face_idx = j
                    break
            
            selected_face = bm.faces[selected_face_idx]
            verts = selected_face.verts
            
            if len(verts) >= 3:
                # generate point using barycentric coordinates
                r1, r2 = random.random(), random.random()
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                r3 = 1 - r1 - r2
                
                local_point = (r1 * verts[0].co + r2 * verts[1].co + r3 * verts[2].co)
                candidate = matrix_world @ local_point
            else:
                candidate = matrix_world @ verts[0].co
            
            if not points:
                # first point, just use it
                best_candidate = candidate
                break
            
            # find minimum distance to existing points
            min_dist_to_existing = float('inf')
            for existing_point in points:
                # manual distance calculation to avoid vector operations
                dx = candidate[0] - existing_point[0]
                dy = candidate[1] - existing_point[1] 
                dz = candidate[2] - existing_point[2]
                dist_squared = dx*dx + dy*dy + dz*dz
                min_dist_to_existing = min(min_dist_to_existing, dist_squared)
            
            # if this candidate is farther from all existing points, use it
            if min_dist_to_existing > best_distance:
                best_distance = min_dist_to_existing
                best_candidate = candidate
                
                # if we found a point that meets minimum distance, use it
                if best_distance >= min_distance * min_distance:  # compare squared distances
                    break
        
        if best_candidate is not None:
            # convert to tuple to avoid keeping vector references
            points.append((best_candidate[0], best_candidate[1], best_candidate[2]))
    
    # critical: free bmesh to prevent memory leak
    bm.free()
    
    # convert back to vectors for return
    return [mathutils.Vector(p) for p in points]

def create_particle_system(obj, particle_count, particle_size):
    """high-performance particle system using dupliverts"""
    # create master particle (single sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=particle_size, location=(0, 0, 0), ring_count=6)
    master_particle = bpy.context.active_object
    master_particle.name = f"{obj.name}_MasterParticle"
    
    # create a mesh with vertices at particle positions
    mesh = bpy.data.meshes.new(f"{obj.name}_ParticlePositions")
    particle_verts = []
    
    # generate evenly distributed particle positions on mesh surface
    # calculate minimum distance based on mesh size and particle count
    mesh_bounds = obj.bound_box
    mesh_size = max(
        abs(mesh_bounds[6][0] - mesh_bounds[0][0]),  # width
        abs(mesh_bounds[6][1] - mesh_bounds[0][1]),  # height
        abs(mesh_bounds[6][2] - mesh_bounds[0][2])   # depth
    )
    
    # adaptive minimum distance based on mesh size and particle density
    min_distance = mesh_size / (particle_count ** 0.5) * 0.8  # 0.8 factor for some overlap tolerance
    
    # generate evenly distributed points
    distributed_points = generate_evenly_distributed_points_on_mesh(obj, particle_count, min_distance)
    
    # build particle data
    distortion_offsets = []
    collision_avoidance_offsets = []
    for i, surface_point in enumerate(distributed_points):
        particle_verts.append(surface_point)
        # pre-generate consistent distortion offset for this particle
        distortion_offset = generate_particle_distortion_offset(i)
        distortion_offsets.append(distortion_offset)
        # pre-generate collision avoidance offset (will be calculated later with target positions)
        collision_avoidance_offsets.append((0.0, 0.0, 0.0))
    
    # create mesh with just vertices (no edges or faces)
    mesh.from_pydata(particle_verts, [], [])
    mesh.update()
    
    # create object from mesh
    particle_positions_obj = bpy.data.objects.new(f"{obj.name}_ParticlePositions", mesh)
    bpy.context.collection.objects.link(particle_positions_obj)
    
    # set up dupliverts on the positions object
    particle_positions_obj.instance_type = 'VERTS'
    particle_positions_obj.use_instance_vertices_rotation = False
    
    # parent master particle to positions object for dupliverts
    master_particle.parent = particle_positions_obj
    
    # create parent empty for the entire system
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    system_parent = bpy.context.active_object
    system_parent.name = f"{obj.name}_ParticleSystem_Controller"
    
    # parent the positions object to the controller
    particle_positions_obj.parent = system_parent
    
    return {
        'master_particle': master_particle,
        'system_parent': system_parent,
        'positions_object': particle_positions_obj,
        'particle_positions': particle_verts,
        'distortion_offsets': distortion_offsets,
        'collision_avoidance_offsets': collision_avoidance_offsets,
        'trajectories': []  # will be populated when trajectories are created
    }

def create_trajectory_curves(particle_system, src_particle_positions, tgt_particle_positions, 
                           src_obj, tgt_obj, distortion_strength, cross_paths, trajectory_resolution=50):
    """curve objects showing particle trajectories"""
    trajectories = []
    
    if not src_particle_positions or not tgt_particle_positions:
        return trajectories
    
    # get the collision avoidance offsets
    collision_avoidance_offsets = particle_system.get('collision_avoidance_offsets', [])
    distortion_offsets = particle_system.get('distortion_offsets', [])
    
    # pre-calculate matrices
    src_matrix = src_obj.matrix_world
    tgt_matrix = tgt_obj.matrix_world
    
    for i in range(len(src_particle_positions)):
        if i >= len(tgt_particle_positions):
            break
            
        # sample points along the trajectory
        trajectory_points = []
        
        for sample in range(trajectory_resolution + 1):
            t = sample / trajectory_resolution
            one_minus_t = 1.0 - t
            
            # get source and target positions in local space
            src_local = src_particle_positions[i]
            tgt_local = tgt_particle_positions[i]
            
            # transform to world space
            src_world_x = src_matrix[0][0] * src_local[0] + src_matrix[0][1] * src_local[1] + src_matrix[0][2] * src_local[2] + src_matrix[0][3]
            src_world_y = src_matrix[1][0] * src_local[0] + src_matrix[1][1] * src_local[1] + src_matrix[1][2] * src_local[2] + src_matrix[1][3]
            src_world_z = src_matrix[2][0] * src_local[0] + src_matrix[2][1] * src_local[1] + src_matrix[2][2] * src_local[2] + src_matrix[2][3]
            
            tgt_world_x = tgt_matrix[0][0] * tgt_local[0] + tgt_matrix[0][1] * tgt_local[1] + tgt_matrix[0][2] * tgt_local[2] + tgt_matrix[0][3]
            tgt_world_y = tgt_matrix[1][0] * tgt_local[0] + tgt_matrix[1][1] * tgt_local[1] + tgt_matrix[1][2] * tgt_local[2] + tgt_matrix[1][3]
            tgt_world_z = tgt_matrix[2][0] * tgt_local[0] + tgt_matrix[2][1] * tgt_local[1] + tgt_matrix[2][2] * tgt_local[2] + tgt_matrix[2][3]
            
            # linear interpolation
            base_x = src_world_x * one_minus_t + tgt_world_x * t
            base_y = src_world_y * one_minus_t + tgt_world_y * t
            base_z = src_world_z * one_minus_t + tgt_world_z * t
            
            # apply distortion if enabled
            if distortion_strength > 0 and i < len(distortion_offsets):
                distortion_offset = distortion_offsets[i]
                distortion_falloff = 4.0 * t * one_minus_t  # parabolic falloff
                distortion_strength_scaled = distortion_falloff * distortion_strength * 0.2
                
                final_x = base_x + distortion_offset[0] * distortion_strength_scaled
                final_y = base_y + distortion_offset[1] * distortion_strength_scaled
                final_z = base_z + distortion_offset[2] * distortion_strength_scaled
            else:
                final_x = base_x
                final_y = base_y
                final_z = base_z
            
            # apply collision avoidance if enabled
            if not cross_paths and i < len(collision_avoidance_offsets):
                avoidance_falloff = 4.0 * t * one_minus_t
                avoidance = collision_avoidance_offsets[i]
                final_x += avoidance[0] * avoidance_falloff
                final_y += avoidance[1] * avoidance_falloff
                final_z += avoidance[2] * avoidance_falloff
            
            trajectory_points.append((final_x, final_y, final_z))
        
        # create curve from trajectory points
        curve_data = bpy.data.curves.new(f"Trajectory_{i}", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2
        
        # create spline
        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(trajectory_points) - 1)  # -1 because one point exists by default
        
        # set points
        for j, point in enumerate(trajectory_points):
            spline.points[j].co = (point[0], point[1], point[2], 1.0)  # x, y, z, weight
        
        # create curve object
        curve_obj = bpy.data.objects.new(f"Trajectory_{i}", curve_data)
        bpy.context.collection.objects.link(curve_obj)
        
        # set curve properties for better visualization
        curve_data.bevel_depth = 0.005  # thin line
        curve_data.bevel_resolution = 2
        
        # create material for trajectory
        mat = bpy.data.materials.new(name=f"TrajectoryMat_{i}")
        mat.use_nodes = True
        
        # set up material nodes for emission
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # add emission node
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value = 0.1
        
        # vary color based on particle index for better visualization
        import colorsys
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for nice color distribution
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        emission.inputs['Color'].default_value = (*rgb, 1.0)
        
        # add output node
        output = nodes.new('ShaderNodeOutputMaterial')
        
        # link nodes
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        # assign material to curve
        curve_obj.data.materials.append(mat)
        
        trajectories.append(curve_obj)
    
    return trajectories

# ensure handler registered only once
def ensure_handler():
    if frame_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frame_handler)

@persistent
def frame_handler(scene):
    frame = scene.frame_current
    to_remove = []
    for name, data in MORPH_DATA.items():
        try:
            obj = bpy.data.objects.get(name)
            if not obj or obj.type != 'MESH':
                to_remove.append(name)
                continue
                
            # check if object data still exists
            if not obj.data:
                to_remove.append(name)
                continue
                
            start = data['start']
            end = data['end']
            src_co = data['src_co']
            tgt_co = data['tgt_co']
            count = len(src_co)
            if frame < start:
                t = 0.0
            elif frame > end:
                t = 1.0
            else:
                t = (frame - start) / (end - start)
            # interpolate vertices
            mesh = obj.data
            for i, v in enumerate(mesh.vertices):
                sc = src_co[i]
                tc = tgt_co[i]
                v.co.x = sc[0] * (1 - t) + tc[0] * t
                v.co.y = sc[1] * (1 - t) + tc[1] * t
                v.co.z = sc[2] * (1 - t) + tc[2] * t
            mesh.update()
            
            # update particles if they exist
            particle_system = data.get('particle_system', {})
            if particle_system:
                src_particle_positions = data.get('src_particle_positions', [])
                tgt_particle_positions = data.get('tgt_particle_positions', [])
                src_obj = data.get('src_obj')
                tgt_obj = data.get('tgt_obj')
                
                # check if source and target objects still exist
                if (src_particle_positions and tgt_particle_positions and 
                    src_obj and tgt_obj and
                    src_obj.name in bpy.data.objects and 
                    tgt_obj.name in bpy.data.objects):
                    
                    try:
                        # get distortion strength and cross_paths setting once
                        distortion_strength = data.get('distortion_strength', 0.0)
                        cross_paths = data.get('cross_paths', True)
                        
                        # direct vertex update without creating intermediate objects
                        positions_object = particle_system.get('positions_object')
                        if positions_object and positions_object.data:
                            mesh = positions_object.data
                            vertices = mesh.vertices
                            
                            # get arrays once (avoid repeated dict lookups)
                            distortion_offsets = particle_system.get('distortion_offsets', [])
                            collision_avoidance_offsets = particle_system.get('collision_avoidance_offsets', [])
                            
                            # pre-calculate matrices and common values once
                            src_matrix = src_obj.matrix_world
                            tgt_matrix = tgt_obj.matrix_world
                            one_minus_t = 1.0 - t
                            
                            # pre-calculate collision avoidance falloff once
                            avoidance_falloff = 0.0
                            if not cross_paths:
                                avoidance_falloff = 4.0 * t * one_minus_t
                            
                            # direct vertex update without intermediate object creation
                            for i, vertex in enumerate(vertices):
                                if i < len(src_particle_positions) and i < len(tgt_particle_positions):
                                    # transform positions directly without vector creation
                                    src_local = src_particle_positions[i]
                                    tgt_local = tgt_particle_positions[i]
                                    
                                    # manual matrix multiplication (faster than vector creation)
                                    src_world_x = src_matrix[0][0] * src_local[0] + src_matrix[0][1] * src_local[1] + src_matrix[0][2] * src_local[2] + src_matrix[0][3]
                                    src_world_y = src_matrix[1][0] * src_local[0] + src_matrix[1][1] * src_local[1] + src_matrix[1][2] * src_local[2] + src_matrix[1][3]
                                    src_world_z = src_matrix[2][0] * src_local[0] + src_matrix[2][1] * src_local[1] + src_matrix[2][2] * src_local[2] + src_matrix[2][3]
                                    
                                    tgt_world_x = tgt_matrix[0][0] * tgt_local[0] + tgt_matrix[0][1] * tgt_local[1] + tgt_matrix[0][2] * tgt_local[2] + tgt_matrix[0][3]
                                    tgt_world_y = tgt_matrix[1][0] * tgt_local[0] + tgt_matrix[1][1] * tgt_local[1] + tgt_matrix[1][2] * tgt_local[2] + tgt_matrix[1][3]
                                    tgt_world_z = tgt_matrix[2][0] * tgt_local[0] + tgt_matrix[2][1] * tgt_local[1] + tgt_matrix[2][2] * tgt_local[2] + tgt_matrix[2][3]
                                    
                                    # interpolate position directly (using pre-calculated one_minus_t)
                                    base_x = src_world_x * one_minus_t + tgt_world_x * t
                                    base_y = src_world_y * one_minus_t + tgt_world_y * t
                                    base_z = src_world_z * one_minus_t + tgt_world_z * t
                                    
                                    # apply distortion if enabled (inline for performance)
                                    if distortion_strength > 0 and i < len(distortion_offsets):
                                        # inline distortion calculation to avoid function call overhead
                                        distortion_offset = distortion_offsets[i]
                                        distortion_falloff = 4.0 * t * one_minus_t  # parabolic falloff
                                        distortion_strength_scaled = distortion_falloff * distortion_strength * 0.2
                                        
                                        final_x = base_x + distortion_offset[0] * distortion_strength_scaled
                                        final_y = base_y + distortion_offset[1] * distortion_strength_scaled
                                        final_z = base_z + distortion_offset[2] * distortion_strength_scaled
                                    else:
                                        final_x = base_x
                                        final_y = base_y
                                        final_z = base_z
                                    
                                    # apply collision avoidance if enabled (using pre-calculated falloff)
                                    if avoidance_falloff > 0 and i < len(collision_avoidance_offsets):
                                        avoidance = collision_avoidance_offsets[i]
                                        final_x += avoidance[0] * avoidance_falloff
                                        final_y += avoidance[1] * avoidance_falloff
                                        final_z += avoidance[2] * avoidance_falloff
                                    
                                    vertex.co.x = final_x
                                    vertex.co.y = final_y
                                    vertex.co.z = final_z
                            
                            # single mesh update (only update if we actually modified vertices)
                            if len(vertices) > 0:
                                mesh.update()
                                
                                # force garbage collection periodically to prevent memory buildup
                                if scene.frame_current % 60 == 0:  # every 60 frames (2 seconds at 30fps)
                                    import gc
                                    gc.collect()
                                    
                            # update trajectory curves in real-time if they exist
                            trajectories = particle_system.get('trajectories', [])
                            if trajectories and src_particle_positions and tgt_particle_positions:
                                try:
                                    # get current object matrices
                                    src_matrix = src_obj.matrix_world
                                    tgt_matrix = tgt_obj.matrix_world
                                    
                                    # get settings from stored data
                                    distortion_strength = data.get('distortion_strength', 0.0)
                                    cross_paths = data.get('cross_paths', True)
                                    distortion_offsets = particle_system.get('distortion_offsets', [])
                                    collision_avoidance_offsets = particle_system.get('collision_avoidance_offsets', [])
                                    
                                    # update each trajectory curve
                                    for i, trajectory_obj in enumerate(trajectories):
                                        if (i < len(src_particle_positions) and 
                                            i < len(tgt_particle_positions) and 
                                            trajectory_obj and trajectory_obj.data):
                                            
                                            curve_data = trajectory_obj.data
                                            if curve_data.splines:
                                                spline = curve_data.splines[0]
                                                points = spline.points
                                                
                                                # get source and target positions for this particle
                                                src_local = src_particle_positions[i]
                                                tgt_local = tgt_particle_positions[i]
                                                
                                                # update each point along the trajectory
                                                for j, point in enumerate(points):
                                                    t = j / (len(points) - 1) if len(points) > 1 else 0.0
                                                    one_minus_t = 1.0 - t
                                                    
                                                    # transform to world space
                                                    src_world_x = src_matrix[0][0] * src_local[0] + src_matrix[0][1] * src_local[1] + src_matrix[0][2] * src_local[2] + src_matrix[0][3]
                                                    src_world_y = src_matrix[1][0] * src_local[0] + src_matrix[1][1] * src_local[1] + src_matrix[1][2] * src_local[2] + src_matrix[1][3]
                                                    src_world_z = src_matrix[2][0] * src_local[0] + src_matrix[2][1] * src_local[1] + src_matrix[2][2] * src_local[2] + src_matrix[2][3]
                                                    
                                                    tgt_world_x = tgt_matrix[0][0] * tgt_local[0] + tgt_matrix[0][1] * tgt_local[1] + tgt_matrix[0][2] * tgt_local[2] + tgt_matrix[0][3]
                                                    tgt_world_y = tgt_matrix[1][0] * tgt_local[0] + tgt_matrix[1][1] * tgt_local[1] + tgt_matrix[1][2] * tgt_local[2] + tgt_matrix[1][3]
                                                    tgt_world_z = tgt_matrix[2][0] * tgt_local[0] + tgt_matrix[2][1] * tgt_local[1] + tgt_matrix[2][2] * tgt_local[2] + tgt_matrix[2][3]
                                                    
                                                    # linear interpolation
                                                    base_x = src_world_x * one_minus_t + tgt_world_x * t
                                                    base_y = src_world_y * one_minus_t + tgt_world_y * t
                                                    base_z = src_world_z * one_minus_t + tgt_world_z * t
                                                    
                                                    # apply distortion if enabled
                                                    if distortion_strength > 0 and i < len(distortion_offsets):
                                                        distortion_offset = distortion_offsets[i]
                                                        distortion_falloff = 4.0 * t * one_minus_t
                                                        distortion_strength_scaled = distortion_falloff * distortion_strength * 0.2
                                                        
                                                        final_x = base_x + distortion_offset[0] * distortion_strength_scaled
                                                        final_y = base_y + distortion_offset[1] * distortion_strength_scaled
                                                        final_z = base_z + distortion_offset[2] * distortion_strength_scaled
                                                    else:
                                                        final_x = base_x
                                                        final_y = base_y
                                                        final_z = base_z
                                                    
                                                    # apply collision avoidance if enabled
                                                    if not cross_paths and i < len(collision_avoidance_offsets):
                                                        avoidance_falloff = 4.0 * t * one_minus_t
                                                        avoidance = collision_avoidance_offsets[i]
                                                        final_x += avoidance[0] * avoidance_falloff
                                                        final_y += avoidance[1] * avoidance_falloff
                                                        final_z += avoidance[2] * avoidance_falloff
                                                    
                                                    # update the curve point
                                                    point.co = (final_x, final_y, final_z, 1.0)
                                                
                                                # update the curve - curves update automatically when points change
                                                # no explicit update needed for curve data
                                    
                                except Exception as e:
                                    # if trajectory update fails, continue with particle animation
                                    print(f"Trajectory update error: {e}")
                                    pass
                    except ReferenceError:
                        # Object was deleted, mark for removal
                        to_remove.append(name)
                else:
                    # source or target objects missing, mark for removal
                    to_remove.append(name)
                    
        except ReferenceError:
            # object was deleted, mark for removal
            to_remove.append(name)
        except Exception as e:
            # any other error, mark for removal and print warning
            print(f"Morph error for {name}: {e}")
            to_remove.append(name)
    
    # remove invalid objects
    for name in to_remove:
        MORPH_DATA.pop(name, None)
    # only remove handler if no morph data remains
    if not MORPH_DATA and frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frame_handler)

# compute subdivision to reach at least target_count
def subdivide_to_count(obj, target_count, max_levels=6):
    mesh = obj.data
    for level in range(max_levels + 1):
        if len(mesh.vertices) >= target_count or level == max_levels:
            break
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=False)
        bm.to_mesh(mesh)
        bm.free()

class OBJECT_OT_MorphMeshes(bpy.types.Operator):
    bl_idname = "object.morph_meshes"
    bl_label = "Morph Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        src = scene.morph_source
        tgt = scene.morph_target
        if not src or not tgt:
            self.report({'ERROR'}, "Specify both Source and Target meshes.")
            return {'CANCELLED'}
        if src.type != 'MESH' or tgt.type != 'MESH' or src == tgt:
            self.report({'ERROR'}, "Invalid Source/Target selection.")
            return {'CANCELLED'}
        if scene.end_frame <= scene.start_frame:
            self.report({'ERROR'}, "End Frame must be after Start Frame.")
            return {'CANCELLED'}

        # duplicate source
        new_mesh = src.data.copy()
        morph = bpy.data.objects.new(src.name + "_Morph", new_mesh)
        context.collection.objects.link(morph)
        morph.matrix_world = src.matrix_world.copy()

        # ensure vertex count match: subdivide morph if fewer verts
        tgt_count = len(tgt.data.vertices)
        if len(morph.data.vertices) < tgt_count:
            subdivide_to_count(morph, tgt_count)
        # now counts >= tgt_count; record src_co and tgt_co arrays
        src_co = [tuple(v.co) for v in morph.data.vertices]
        # map target coords by wrapping if fewer
        tgt_co = []
        for i in range(len(src_co)):
            idx = i % tgt_count
            tgt_co.append(tuple(tgt.data.vertices[idx].co))

        # handle particles if enabled
        particle_system = {}
        src_particle_positions = []
        tgt_particle_positions = []
        
        if scene.use_particles:
            # hide the morph object
            morph.hide_viewport = True
            morph.hide_render = True
            
            # create particle system
            particle_system = create_particle_system(morph, scene.particle_count, scene.particle_size)
            
            # store positions in local space relative to source object
            src_particle_positions = []
            for world_pos in particle_system['particle_positions']:
                # convert world position to local space of source object
                local_pos = src.matrix_world.inverted() @ mathutils.Vector(world_pos)
                src_particle_positions.append(tuple(local_pos))
            
            # calculate target positions in local space of target object with even distribution
            particle_count = len(particle_system['particle_positions'])
            
            # calculate minimum distance for target mesh
            tgt_mesh_bounds = tgt.bound_box
            tgt_mesh_size = max(
                abs(tgt_mesh_bounds[6][0] - tgt_mesh_bounds[0][0]),
                abs(tgt_mesh_bounds[6][1] - tgt_mesh_bounds[0][1]),
                abs(tgt_mesh_bounds[6][2] - tgt_mesh_bounds[0][2])
            )
            tgt_min_distance = tgt_mesh_size / (particle_count ** 0.5) * 0.8
            
            # generate evenly distributed target positions
            distributed_tgt_points = generate_evenly_distributed_points_on_mesh(tgt, particle_count, tgt_min_distance)
            
            tgt_particle_positions = []
            for world_tgt_pos in distributed_tgt_points:
                # convert to local space of target object
                local_tgt_pos = tgt.matrix_world.inverted() @ world_tgt_pos
                tgt_particle_positions.append(tuple(local_tgt_pos))
            
            # calculate guaranteed safe collision avoidance offsets if cross_paths is disabled
            if not scene.cross_paths:
                print(f"Calculating guaranteed collision avoidance for {len(src_particle_positions)} particles with distance {scene.avoidance_distance}")
                safe_offsets = calculate_guaranteed_safe_offsets(
                    src_particle_positions,
                    tgt_particle_positions,
                    scene.avoidance_distance
                )
                
                # update the collision avoidance offsets with guaranteed safe values
                collision_avoidance_offsets = particle_system['collision_avoidance_offsets']
                for i, offset in enumerate(safe_offsets):
                    if i < len(collision_avoidance_offsets):
                        collision_avoidance_offsets[i] = offset
                
                print(f"Collision avoidance calculation complete. Maximum offset magnitude: {max([math.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in safe_offsets]):.3f}")

            # create trajectory visualization if enabled
            if scene.show_trajectories:
                trajectories = create_trajectory_curves(
                    particle_system, 
                    src_particle_positions, 
                    tgt_particle_positions,
                    src, 
                    tgt, 
                    scene.distortion_strength,
                    scene.cross_paths,
                    scene.trajectory_resolution
                )
                particle_system['trajectories'] = trajectories
                
                # parent all trajectory curves to the system controller
                for trajectory in trajectories:
                    trajectory.parent = particle_system['system_parent']
            else:
                # create trajectories even without collision avoidance if requested
                if scene.show_trajectories:
                    trajectories = create_trajectory_curves(
                        particle_system, 
                        src_particle_positions, 
                        tgt_particle_positions,
                        src, 
                        tgt, 
                        scene.distortion_strength,
                        scene.cross_paths,
                        scene.trajectory_resolution
                    )
                    particle_system['trajectories'] = trajectories
                    
                    # parent all trajectory curves to the system controller
                    for trajectory in trajectories:
                        trajectory.parent = particle_system['system_parent']

        # store data
        MORPH_DATA[morph.name] = {
            'src_co': src_co,
            'tgt_co': tgt_co,
            'start': scene.start_frame,
            'end':   scene.end_frame,
            'src_obj': src,
            'tgt_obj': tgt,
            'particle_system': particle_system,
            'src_particle_positions': src_particle_positions,
            'tgt_particle_positions': tgt_particle_positions,
            'distortion_strength': scene.distortion_strength if scene.use_particles else 0.0,
            'cross_paths': scene.cross_paths,
        }
        ensure_handler()
        # jump to start frame
        scene.frame_set(scene.start_frame)
        
        if scene.use_particles:
            particle_count = len(particle_system.get('particle_positions', []))
            self.report({'INFO'}, f"Morphing {src.name} → {tgt.name} with {particle_count} particles from frame {scene.start_frame} to {scene.end_frame}")
        else:
            self.report({'INFO'}, f"Morphing {src.name} → {tgt.name} from frame {scene.start_frame} to {scene.end_frame}")
        return {'FINISHED'}

class OBJECT_OT_ClearAnimation(bpy.types.Operator):
    bl_idname = "object.clear_animation"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # safely clear morph data
        MORPH_DATA.clear()
        
        # force remove all instances of frame handler (in case of duplicates)
        handlers_removed = 0
        while frame_handler in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(frame_handler)
            handlers_removed += 1
        
        # force refresh the scene to stop any lingering updates
        context.scene.frame_set(context.scene.frame_current)
        
        # aggressive memory cleanup
        import gc
        gc.collect()
        
        # force update all mesh objects to clear any cached data
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.data:
                try:
                    obj.data.update()
                except:
                    pass
        
        self.report({'INFO'}, f"Cleared all morph animations, removed {handlers_removed} handlers, forced garbage collection")
        return {'FINISHED'}

class OBJECT_OT_DebugMorph(bpy.types.Operator):
    bl_idname = "object.debug_morph"
    bl_label = "Debug Morph System"
    bl_options = {'REGISTER'}

    def execute(self, context):
        # count active morphs
        active_morphs = len(MORPH_DATA)
        
        # count frame handlers
        handler_count = bpy.app.handlers.frame_change_pre.count(frame_handler)
        
        # count potential leftover objects
        morph_objects = 0
        particle_objects = 0
        for obj in bpy.data.objects:
            if "_Morph" in obj.name:
                morph_objects += 1
            if ("_MasterParticle" in obj.name or "_ParticlePositions" in obj.name or "_ParticleSystem_Controller" in obj.name):
                particle_objects += 1
        
        # count potential leftover meshes
        particle_meshes = 0
        for mesh in bpy.data.meshes:
            if "_ParticlePositions" in mesh.name:
                particle_meshes += 1
        
        # count bmesh instances (potential memory leak source)
        import gc
        bmesh_count = len([obj for obj in gc.get_objects() if hasattr(obj, '__class__') and 'bmesh' in str(type(obj))])
        
        # force garbage collection and recount
        gc.collect()
        bmesh_count_after_gc = len([obj for obj in gc.get_objects() if hasattr(obj, '__class__') and 'bmesh' in str(type(obj))])
        
        self.report({'INFO'}, f"Morphs: {active_morphs}, Handlers: {handler_count}, Objects: {morph_objects + particle_objects}, Meshes: {particle_meshes}, BMesh: {bmesh_count}->{bmesh_count_after_gc}")
        print(f"MORPH DEBUG: Active morphs: {active_morphs}")
        print(f"MORPH DEBUG: Frame handlers: {handler_count}")
        print(f"MORPH DEBUG: Morph objects: {morph_objects}")
        print(f"MORPH DEBUG: Particle objects: {particle_objects}")
        print(f"MORPH DEBUG: Particle meshes: {particle_meshes}")
        print(f"MORPH DEBUG: BMesh objects before/after GC: {bmesh_count}/{bmesh_count_after_gc}")
        
        return {'FINISHED'}

class OBJECT_OT_CleanupParticles(bpy.types.Operator):
    bl_idname = "object.cleanup_particles"
    bl_label = "Cleanup Particles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # clear morph data first
        MORPH_DATA.clear()
        
        # force remove all instances of frame handler
        handlers_removed = 0
        while frame_handler in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(frame_handler)
            handlers_removed += 1
        
        # remove all particle system objects and collections
        systems_removed = 0
        collections_removed = 0
        
        # remove objects related to particle systems
        for obj in list(bpy.data.objects):
            if ("_MasterParticle" in obj.name or 
                "_ParticlePositions" in obj.name or 
                "_ParticleSystem_Controller" in obj.name or
                "_Morph" in obj.name or
                "Trajectory_" in obj.name):
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                    systems_removed += 1
                except ReferenceError:
                    # object already deleted, skip
                    pass
        
        # remove particle position meshes and trajectory curves
        for mesh in list(bpy.data.meshes):
            if "_ParticlePositions" in mesh.name:
                try:
                    bpy.data.meshes.remove(mesh)
                    collections_removed += 1
                except ReferenceError:
                    # mesh already deleted, skip
                    pass
        
        # remove trajectory curves and materials
        for curve in list(bpy.data.curves):
            if "Trajectory_" in curve.name:
                try:
                    bpy.data.curves.remove(curve)
                    collections_removed += 1
                except ReferenceError:
                    pass
        
        for material in list(bpy.data.materials):
            if "TrajectoryMat_" in material.name:
                try:
                    bpy.data.materials.remove(material)
                    collections_removed += 1
                except ReferenceError:
                    pass
        
        # force refresh the scene
        context.scene.frame_set(context.scene.frame_current)
        
        # force garbage collection to free memory
        import gc
        gc.collect()
        
        self.report({'INFO'}, f"Removed {systems_removed} objects, {collections_removed} meshes, {handlers_removed} handlers")
        return {'FINISHED'}

class VIEW3D_PT_MorphPanel(bpy.types.Panel):
    bl_label = "Mesh Morph"
    bl_idname = "VIEW3D_PT_mesh_morph"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mesh Morph"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # object selection
        layout.prop(scene, "morph_source")
        layout.prop(scene, "morph_target")
        
        # particle settings  
        layout.separator()
        layout.label(text="Particle Settings:")
        layout.prop(scene, "use_particles")
        
        if scene.use_particles:
            layout.prop(scene, "particle_count")
            layout.prop(scene, "particle_size")
            layout.separator()
            layout.label(text="Distortion:")
            layout.prop(scene, "distortion_strength")
            layout.separator()
            layout.label(text="Collision Avoidance (Drone Safety):")
            layout.prop(scene, "cross_paths")
            if not scene.cross_paths:
                layout.prop(scene, "avoidance_distance")
            layout.separator()
            layout.label(text="Trajectory Visualization:")
            layout.prop(scene, "show_trajectories")
            if scene.show_trajectories:
                layout.prop(scene, "trajectory_resolution")
        
        # animation controls
        layout.separator()
        layout.label(text="Animation Timing:")
        layout.prop(scene, "start_frame")
        layout.prop(scene, "end_frame")
        layout.operator(OBJECT_OT_MorphMeshes.bl_idname, text="Morph Meshes")
        
        # cleanup
        layout.separator()
        layout.operator(OBJECT_OT_ClearAnimation.bl_idname, text="Clear Animation")
        layout.operator(OBJECT_OT_CleanupParticles.bl_idname, text="Cleanup Particles")
        layout.operator(OBJECT_OT_DebugMorph.bl_idname, text="Debug System")


######################
# // registration // #
######################

# store list of classes for easy registration       
_classes = [
    OBJECT_OT_MorphMeshes,
    OBJECT_OT_ClearAnimation,
    OBJECT_OT_DebugMorph,
    OBJECT_OT_CleanupParticles,
    VIEW3D_PT_MorphPanel,
]

def register_properties():
    """register all custom scene properties"""
    Scene.morph_source = bpy.props.PointerProperty(name="Source Object", type=bpy.types.Object)
    Scene.morph_target = bpy.props.PointerProperty(name="Target Object", type=bpy.types.Object)
    Scene.use_particles = bpy.props.BoolProperty(name="Use Particles", default=False, description="Cover objects with particles instead of showing mesh")
    Scene.particle_count = bpy.props.IntProperty(name="Particle Count", default=100, min=1, max=10000, description="Number of particles to create")
    Scene.particle_size = bpy.props.FloatProperty(name="Particle Size", default=0.05, min=0.001, max=1.0, description="Size of each particle sphere")
    Scene.distortion_strength = bpy.props.FloatProperty(name="Distortion Strength", default=0.0, min=0.0, max=20.0, description="Amount of turbulence distortion during morph")
    Scene.cross_paths = bpy.props.BoolProperty(name="Allow Crossing Paths", default=True, description="Allow particles to cross paths")
    Scene.avoidance_distance = bpy.props.FloatProperty(name="Avoidance Distance", default=0.5, min=0.1, max=5.0, description="Minimum distance between particle paths for collision avoidance")
    Scene.show_trajectories = bpy.props.BoolProperty(name="Show Trajectories", default=False, description="Display particle path trajectories as curves")
    Scene.trajectory_resolution = bpy.props.IntProperty(name="Trajectory Resolution", default=50, min=10, max=200, description="Number of points along each trajectory curve")
    Scene.start_frame = bpy.props.IntProperty(name="Start Frame", default=1, min=1, description="Frame where morphing animation begins")
    Scene.end_frame = bpy.props.IntProperty(name="End Frame", default=250, min=1, description="Frame where morphing animation ends")

def unregister_properties():
    """unregister all custom scene properties"""
    del Scene.morph_source
    del Scene.morph_target
    del Scene.use_particles
    del Scene.particle_count
    del Scene.particle_size
    del Scene.distortion_strength
    del Scene.cross_paths
    del Scene.avoidance_distance
    del Scene.show_trajectories
    del Scene.trajectory_resolution
    del Scene.start_frame
    del Scene.end_frame

def register():
    # register properties before registering classes
    register_properties()
     
    for cls in _classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"Failed to register class {cls}: {e}")

def unregister():
    # unregister classes first (in reverse order)
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Failed to unregister class {cls}: {e}")
    
    # unregister properties after unregistering classes
    unregister_properties()

# hot reload, live updates. <-- dev only remove on release
register()
