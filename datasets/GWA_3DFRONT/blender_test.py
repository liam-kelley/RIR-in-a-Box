import bpy

bpy.ops.mesh.primitive_cube_add(size=4)

cube_obj = bpy.context.active_object
cube_obj.location.z = 5