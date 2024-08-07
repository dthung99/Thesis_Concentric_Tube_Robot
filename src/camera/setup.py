from setuptools import find_packages, setup

package_name = 'camera'

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
    maintainer='dangthehung',
    maintainer_email='dangthehung@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'print_hello = camera.print_hello:main',
            'register_camera_and_save = camera.register_camera_and_save:main',
            'segment_tubes = camera.segment_tubes:main',            
            'test = camera.test:main',
        ],
    },
)
