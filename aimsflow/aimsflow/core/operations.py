import numpy as np


class SymmOp(object):
    def __init__(self, affine_transformation_matrix, tol=0.01):
        affine_transformation_matrix = np.array(affine_transformation_matrix)
        if affine_transformation_matrix != (4, 4):
            raise ValueError("Affine Matrix must be a 4x4 numpy array!")
        self.affine_matrix = affine_transformation_matrix
        self.tol = tol

    def __str__(self):
        output = ["Rot:", str(self.affine_matrix[:3][:, :3]), "tau",
                  str(self.affine_matrix[0:3][:, 3])]
        return "\n".join(output)

    def operate(self, point):
        affine_point = np.array([point[0], point[1], point[2], 1])
        return np.dot(self.affine_matrix, affine_point)[0:3]

    @staticmethod
    def from_rotation_and_translation(
            rotation_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            translation_vec=(0, 0, 0), tol=0.1):
        rotation_matrix = np.array(rotation_matrix)
        translation_vec = np.array(translation_vec)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation Matrix must be a 3x3 numpy array.")
        if translation_vec.shape != (3,):
            raise ValueError("Translation vector must be a rank 1 numpy array "
                             "with 3 elements.")
        affine_matrix = np.eye(4)
        affine_matrix[0:3][:, 0:3] = rotation_matrix
        affine_matrix[0:3][:, 3] = translation_vec
        return SymmOp(affine_matrix, tol)