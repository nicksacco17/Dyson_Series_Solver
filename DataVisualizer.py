print("VISUALIZATION")
    plt.title('Overlap between QME and AQC')
    fig = plt.figure()
    fig, ax = qp.visualization.hinton(H_PROB.my_Hamiltonian)
    
    print("DONE")
    plt.show()