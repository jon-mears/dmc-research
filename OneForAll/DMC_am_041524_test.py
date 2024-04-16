import DMC_am_041524 as code

if __name__ == '__main__':
    particles2mass = {'H': 1.007825, 'O': 15.99491461957}
    molecules2counts = {'H2O': 20, 'H3O2': 50}

    w1 = code.Walkers(particles2mass, molecules2counts)

    assert w1.nMolecules == 70
    assert w1.nParticles == 5
    assert w1.mlclidx['H2O'] == list(range(0, 20))
    assert w1.mlclidx['H3O2'] == list(range(20, 70))
    assert w1.ptclidx['H2O H'] == list(range(0, 2))
    assert w1.ptclidx['H2O O'] == [2]
    assert w1.ptclidx['H3O2 H'] == list(range(0, 3))
    assert w1.ptclidx['H3O2 O'] == list(range(3, 5))

    particles2mass = {'Ne': 1, 'I': 1, 'He': 1, 'Cl': 1, 'C': 1}
    molecules2counts = {'I2Ne4He3': 25, 'CCl4Ne': 21}

    w2 = code.Walkers(particles2mass, molecules2counts)

    assert w2.nMolecules == 46
    assert w2.nParticles == 9
    assert w2.mlclidx['I2Ne4He3'] == list(range(0, 25))
    assert w2.mlclidx['CCl4Ne'] == list(range(25, 46))
    assert w2.ptclidx['I2Ne4He3 I'] == list(range(0, 2))
    assert w2.ptclidx['I2Ne4He3 Ne'] == list(range(2, 6))
    assert w2.ptclidx['I2Ne4He3 He'] == list(range(6, 9))
    assert w2.ptclidx['CCl4Ne C'] == [0]
    assert w2.ptclidx['CCl4Ne Cl'] == list(range(1, 5))
    assert w2.ptclidx['CCl4Ne Ne'] == [5]