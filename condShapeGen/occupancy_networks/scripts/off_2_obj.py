import os
import meshio



def src_to_dst(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in os.listdir(input_folder):
        print(file)
        if file.endswith('.off') and '_rtv' not in file and '_gt' not in file:
            input_path = os.path.join(input_folder, file)
            mesh = meshio.read(
                input_path, 
            )
            output_path = os.path.join(output_folder, file.replace('.off', '.obj'))

            mesh.write(
                output_path
            )

if __name__ == '__main__':
    input_path = os.path.join('demos/free-formed')
    output_path = os.path.join('demos/free-formed-obj')
    src_to_dst(input_path, output_path)
