import yaml
import argparse

def read_yaml(yaml_path):
    data_ann = None
    with open(yaml_path, 'r') as stream:
        try:
            data_ann = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return data_ann

def parse_opt(ROOT, known=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default=ROOT /
                        'data.yaml', help='dataset.yaml path relative to datasets folder')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='total training epochs')

    parser.add_argument('--use_comet_ml', type=bool, default=False,
                        help='Use comet ml to post results')

    return parser.parse_known_args()[0] if known else parser.parse_args()