from ...ui.research_panel import ResearchPanel

class CatNetWidget(ResearchPanel):
    def __init__(self,filename,image,parent=None):
        super().__init__(image,'catnet',['CAT-Net v2'],
            'Analyse RGB et coefficients JPEG. JPEG : fichier original ; autres formats : compagnon JPEG 100, sans modifier la source.',parent,filename=filename)
