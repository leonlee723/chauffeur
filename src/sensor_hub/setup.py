from setuptools import setup

package_name = 'sensor_hub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name],),
        ('share/' + package_name, ['package.xml']),
        ('share/'+ package_name +'/meshes',['resource/4096-MicroWheelsG.dae'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='leonlee723@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = sensor_hub.kitti_pub:main',
        ],
    },
)
