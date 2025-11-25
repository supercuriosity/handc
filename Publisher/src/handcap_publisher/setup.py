from setuptools import find_packages, setup

package_name = 'handcap_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='hong_li@sjtu.edu.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'sum_node = handcap_publisher.sum_node:main',
            'camera_node = handcap_publisher.camera_node:main',
            'left_tactile_node = handcap_publisher.left_tactile_node:main',
            'right_tactile_node = handcap_publisher.right_tactile_node:main',
            'left_force_node = handcap_publisher.left_force_node:main',
            'right_force_node = handcap_publisher.right_force_node:main',
            'angle_node = handcap_publisher.angle_node:main',
        ],
    },
)
