import json
import subprocess

# ami_id = "ami-0c7a4d8f0a1256849"
ami_id = "ami-0852df70b05b846ea" # With AWS CLI
instance_type = 'p3.2xlarge'
key_name = 'Assaf'
secutiry_groups = "launch-wizard-10"

max_price = '1.5'
market_type_dict = {"MarketType": "spot", "SpotOptions": {"MaxPrice": max_price, "SpotInstanceType": "persistent",
                                                          "InstanceInterruptionBehavior": "stop"}}
market_type_json = json.dumps(market_type_dict)
run_3d = False
if run_3d:
    user_data = "#!/bin/bash\\n" \
                "cp -u -r -v /media/efs/CellTrackingChallenge/Training/{dataset} /home/ubuntu/\\n" \
                "python3 /home/ubuntu/lstm-unet/train3DSlice.py --aws " \
                "--experiment_name {experiment_name} --gpu_id 0" \
                " --root_data_dir '/home/ubuntu/' " \
                "--save_log_dir '/media/efs/Outputs' --dataset {dataset_string} " \
                "--val_dataset {val_dataset_string} --depth_pad {depth_pad} --erode_dilate_tra {erode_dilate_tra0} " \
                "{erode_dilate_tra1} "\
                "--add_tra_output --num_iterations 500000"
    data_to_run = [
        # ('Fluo-C3DL-MDA231', 3),
        ('Fluo-C3DH-H157', (3,0)),
        ('Fluo-N3DH-CE', (7,0)),
        # ('Fluo-N3DH-CHO', 3),
        # ('Fluo-N3DH-SIM+', -11),
        # ('Fluo-C3DH-A549', 3),
        # ('Fluo-C3DH-A549-SIM', -11),
        # ('Fluo-N3DL-TRIC', (3, 0)),
        # ('Fluo-N3DL-DRO', (3, 0)),
    ]
else:
    user_data = "#!/bin/bash\\npython3 /home/ubuntu/lstm-unet/train2D.py --aws " \
                "--experiment_name {experiment_name} --gpu_id 0" \
                " --root_data_dir '/media/efs/CellTrackingChallenge/Training/' " \
                "--save_log_dir '/media/efs/Outputs' --dataset {dataset_string} " \
                "--val_dataset {val_dataset_string} --erode_dilate_tra {erode_dilate_tra0}  {erode_dilate_tra1} "\
                "--add_tra_output --num_iterations 300000"

    data_to_run = [
        # ('Fluo-N2DH-GOWT1', 0),
       # ('Fluo-N2DL-HeLa', (5, -5)),
        ('DIC-C2DH-HeLa', (15, -15))#for 3
        # ('PhC-C2DL-PSC', (1, -2)),
        # ('Fluo-C2DL-MSC', 0),
        # ('PhC-C2DH-U373', 3),
        # ('Fluo-N2DH-SIM+', -9),
        # ('Fluo-N2DH-MYSIMV3', -10)
    ]
for dataset_name, erode_dilate_tra in data_to_run:
    experiment_name = dataset_name + 'SegTra'
    depth_pad = 3
    # erode_dilate_tra = 0

    dataset_string_template = '{dataset_name} '.format(dataset_name=dataset_name) + '{:02d} '
    dataset_string = ''
    for s in range(1, 3):
    # for s in range(1, 45):
        dataset_string += dataset_string_template.format(s)
    val_dataset_string = ''
    for s in range(1, 3):
    # for s in range(45, 52):
        val_dataset_string += dataset_string_template.format(s)

    if run_3d:
        full_user_data = user_data.format(experiment_name=experiment_name, dataset_string=dataset_string,
                                          val_dataset_string=val_dataset_string,
                                          depth_pad=depth_pad, erode_dilate_tra0=erode_dilate_tra[0],
                                          erode_dilate_tra1=erode_dilate_tra[1], dataset=dataset_name)
    else:
        full_user_data = user_data.format(experiment_name=experiment_name, dataset_string=dataset_string,
                                          val_dataset_string=val_dataset_string,
                                          erode_dilate_tra0=erode_dilate_tra[0], erode_dilate_tra1=erode_dilate_tra[1])
    tag_dict = [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": experiment_name}]},
                ]
    tag_json = json.dumps(tag_dict)
    launch_command = ["aws", "ec2", "run-instances", "--image-id", "{ami_id}".format(ami_id=ami_id),
                      "--instance-type", "{instance_type}".format(instance_type=instance_type), "--key-name",
                      "{key_name}".format(key_name=key_name), "--security-groups",
                      "{secutiry_groups}".format(secutiry_groups=secutiry_groups), "--user-data",
                      "{user_data}".format(user_data=full_user_data),
                      "--instance-market-options", "{market_type}".format(market_type=market_type_json),
                      "--tag-specifications",
                      "{tag_string}".format(tag_string=tag_json)]

    result = subprocess.run(launch_command, stdout=subprocess.PIPE)
    result_dict = json.loads(result.stdout.decode('utf-8'))
    print(result_dict)
    tag_spot_command = ["aws", "ec2", "create-tags", "--resources",
                        result_dict['Instances'][0]['SpotInstanceRequestId'],
                        "--tags", 'Key="Name",Value="{}"'.format(experiment_name)]
    result = subprocess.run(tag_spot_command, stdout=subprocess.PIPE)
    # result_dict = json.loads(result.stdout.decode('utf-8'))
    print(result.stdout.decode('utf-8'))
    # print(launch_command)
# for d in data_to_run:
#     print(d[0])
import awscli
