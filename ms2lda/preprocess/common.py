class MS1(object):
    def __init__(self, id, mz, rt, intensity, file_name, scan_number=None, single_charge_precursor_mass=None):
        self.id = id
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.file_name = file_name
        self.scan_number = scan_number
        if single_charge_precursor_mass:
            self.single_charge_precursor_mass = single_charge_precursor_mass
        else:
            self.single_charge_precursor_mass = self.mz
        self.name = "{}_{}".format(self.mz, self.rt)

    def __str__(self):
        return self.name
