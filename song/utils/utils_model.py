import torch


def save_model(model, optimizer, path, **kwargs):
    dict_save = {'model_state_dict': model.state_dict()}
    if optimizer is None:
        dict_save['optimizer_state_dict'] = optimizer
    else:
        dict_save['optimizer_state_dict'] = optimizer.state_dict()
    dict_save.update(kwargs)
    torch.save(dict_save, path)


def load_model(model, optimizer, path, device):
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    del checkpoint['optimizer_state_dict'], checkpoint['model_state_dict']

    model.eval()
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, optimizer, checkpoint


def load_model_selected_graph(model, path, id_graph, device):
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    if id_graph is not None:
        for key, val in checkpoint['model_state_dict'].items():
            if key.startswith("graph"):
                if val.dim() > 1:
                    checkpoint['model_state_dict'][key] = val.data[id_graph, ...].unsqueeze(0)
                else:
                    checkpoint['model_state_dict'][key] = val.data
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    del checkpoint['model_state_dict']
    checkpoint['acc'] = checkpoint['acc'][id_graph]
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, checkpoint


def load_selected_weights(model, path, id_graph, no_resume_weights, device):
    assert set(no_resume_weights).issubset(['representation', 'm', 'nodes', 'root'])
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)

    using_state_dict = {}
    for key, val in checkpoint['model_state_dict'].items():
        if 'representation' not in no_resume_weights and key.startswith('representation'):
            using_state_dict[key] = val
        elif 'm' not in no_resume_weights and (key.startswith('graph.M_left') or key.startswith('graph.M_right')):
            using_state_dict[key] = val.data[id_graph, ...].unsqueeze(0) if id_graph is not None and val.dim() > 1 else val
        elif 'nodes' not in no_resume_weights and key.startswith('graph.nodes'):
            using_state_dict[key] = val.data[id_graph, ...].unsqueeze(0) if id_graph is not None and val.dim() > 1 else val
        elif 'root' not in no_resume_weights and key.startswith('graph.roots'):
            using_state_dict[key] = val.data[id_graph, ...].unsqueeze(0) if id_graph is not None and val.dim() > 1 else val

    del checkpoint['model_state_dict']

    model_dict = model.state_dict()
    model_dict.update(using_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    checkpoint['acc'] = checkpoint['acc'][id_graph]
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, checkpoint
