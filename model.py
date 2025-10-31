
import torch
from LYTNetV2 import LYTNetV2

class MakeModel(torch.nn.Module):
    def __init__(self, pretrained_state_dict_path='moreefficientweights.pth', pretrained=True, freeze=False):
        super(MakeModel, self).__init__()
        self.model = LYTNetV2()
        if pretrained:
            try:
                self.model.load_state_dict(torch.load(pretrained_state_dict_path, map_location=torch.device('cpu')))
            except:
                raise AssertionError
        self.model = self.model.features
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.IScrw_output = torch.nn.Sequential(torch.nn.Linear(1280, 1), torch.nn.Sigmoid())
        self.coord_output = torch.nn.Sequential(torch.nn.Linear(1280, 4))  # Always define it

    def forward(self, x):
        y = self.model(x).view(-1, 1280)
        is_crosswalk = self.IScrw_output(y)  # Compute crosswalk probability

        # Only compute coordinates if crosswalk probability is high
        coordinates = self.coord_output(y) * (is_crosswalk >= 0.5).float()

        return {'coordinates': coordinates, 'IScrosswalk': is_crosswalk}
