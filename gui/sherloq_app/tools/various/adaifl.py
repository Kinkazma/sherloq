from ...ui.research_panel import ResearchPanel

class AdaIFLWidget(ResearchPanel):
    def __init__(self,image,parent=None):
        super().__init__(image,'adaifl',['AdaIFL v0'],
            'Localisation apprise à 1024 × 1024 px ; carte continue et masque au seuil 0,5.',parent)
        self.cpu.setChecked(True)
