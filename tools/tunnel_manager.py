import os
import json

class Tunnel_Manager:
    def __init__(self):
        self.objects = {}

    def get_tunnel_data(self, tunnel):
        tunnel_data_folder = r'.\\data\\tunnel_data'
        tunnel_data_file = os.path.join(tunnel_data_folder, tunnel)
        tunnel_data_file += '.json'
        if not os.path.exists(tunnel_data_file):
            return False

        with open(tunnel_data_file, 'r') as f:
            tunnel_data = json.load(f)

        return tunnel_data