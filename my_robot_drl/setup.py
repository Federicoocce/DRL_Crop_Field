from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'my_robot_drl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line installs your models
        (os.path.join('share', package_name, 'models', 'waypoint_sphere'), glob(os.path.join('models', 'waypoint_sphere', '*'))),
        # --- THIS IS THE MISSING LINE ---
        # This line tells the build system to find all .launch.py files
        # in your 'launch' directory and install them.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fede',
    maintainer_email='fede@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'field_data_reader = my_robot_drl.get_field_data:main',
        'waypoint_visualizer = my_robot_drl.waypoint_visualizer:main',
        'train_agent = my_robot_drl.train_agent:main',
        ],
    },
)