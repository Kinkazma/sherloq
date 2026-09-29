from ...ui.research_panel import ResearchPanel

class FocalWidget(ResearchPanel):
    def __init__(self,image,parent=None):
        super().__init__(image,'focal',['ViT-L + HRNet'],
            'Partition en deux groupes à 1024 × 1024 px ; le plus petit est affiché comme candidat. Une partition ne prouve pas une retouche.',parent)
