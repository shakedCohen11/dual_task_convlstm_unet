import requests
import subprocess
import json


def get_instance_id():
    return requests.get(('http://169.254.169.254/latest/meta-data/instance-id')).text


def get_spot_id(instance_id):
    get_spot_command = ['aws', 'ec2',  "describe-spot-instance-requests", "--filters",
                        "Name=instance-id,Values={}".format(instance_id)]
    result = subprocess.run(get_spot_command, stdout=subprocess.PIPE)
    result_dict = json.loads(result.stdout.decode('utf-8'))
    spot_id = result_dict['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    return spot_id


def terminate_spot_request(spot_id):
    terminate_command = ['aws', 'ec2', 'cancel-spot-instance-requests', "--spot-instance-request-ids", spot_id]
    result = subprocess.run(terminate_command, stdout=subprocess.PIPE)
    print((result.stdout.decode('utf-8')))


def terminate_instance(instance_id):
    terminate_command = ['aws', 'ec2',  'terminate-instances', '--instance-ids', instance_id]
    result = subprocess.run(terminate_command, stdout=subprocess.PIPE)
    print((result.stdout.decode('utf-8')))


def stop_this_instance():

    instance_id = get_instance_id()
    print('Got instance ID: {}'.format(instance_id))
    spot_id = get_spot_id(instance_id)
    print('Got Spot ID: {}'.format(spot_id))
    terminate_spot_request(spot_id)
    print('Stopping Spot instance: {}'.format(spot_id))
    terminate_instance(instance_id)
    print('Stopping instance: {}'.format(instance_id))


def should_spot_terminate():
    r = requests.get('http://169.254.169.254/latest/meta-data/spot/instance-action')
    if not r.status_code == 404:
        return True
    else:
        return False
