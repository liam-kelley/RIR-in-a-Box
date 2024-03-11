import numpy as np

def edge_matrix_from_face_matrix(face_matrix):
    '''
    DEPRECATED
    '''
    # Iterate over face_matrix to extract edges
    edges = []
    for f in face_matrix:
        face_edges = [(f[i], f[(i+1) % len(f)]) for i in range(len(f))]
        for e in face_edges:
            if e not in edges and (e[1], e[0]) not in edges:  # Avoid duplicates
                edges.append(e)
    # Convert edges list to a numpy array
    edge_matrix = np.array(edges)
    return edge_matrix