import sys
from PyQt5.QtWidgets import QApplication, QScrollArea, QWidget, QLabel, QGridLayout, QLineEdit, QPushButton, QVBoxLayout, QListView, QComboBox, QListWidget, QGroupBox, QFileDialog, QCheckBox
from PyQt5.QtGui import QStandardItemModel, QStandardItem

class ConfigGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Configuration File Generator')
        self.showMaximized()
        

        self.main_widget = QWidget()  # Create a widget to hold your layout
        self.main_layout = QGridLayout(self.main_widget)



        ## EXPERIMENT ##
        self.experiment_groupbox = QGroupBox('Experiment')
        experiment_layout = QVBoxLayout()
        self.experiment_groupbox.setLayout(experiment_layout)
        
        # EXPERIMENT NUMBER
        self.experiment_number_label = QLabel('Experiment number')
        experiment_layout.addWidget(self.experiment_number_label)
        self.experiment_number_edit = QLineEdit()
        experiment_layout.addWidget(self.experiment_number_edit)

        # EXPERIMENT TYPE
        self.experiment_type_label = QLabel('Experiment type')
        experiment_layout.addWidget(self.experiment_type_label)
        self.experiment_type_combo = QComboBox()
        self.experiment_type_combo.addItems(["Mutagenese", "Genome", "Simulation"])
        experiment_layout.addWidget(self.experiment_type_combo)

        self.experiment_type_combo.currentIndexChanged.connect(self.handleExperimentTypeChange)


        self.main_layout.addWidget(self.experiment_groupbox, 0, 0)



        ## PATHS ##
        self.paths_groupbox = QGroupBox('Paths')
        paths_layout = QVBoxLayout()
        self.paths_groupbox.setLayout(paths_layout)

        # HOME DIRECTORY
        self.home_dir_label = QLabel("Home directory:")
        paths_layout.addWidget(self.home_dir_label)
        
        self.home_dir_selected_label = QLabel()
        paths_layout.addWidget(self.home_dir_selected_label)
        
        self.select_home_directory_button = QPushButton('Select Directory')
        self.select_home_directory_button.clicked.connect(lambda: self.open_directory_dialog(self.home_dir_selected_label))
        paths_layout.addWidget(self.select_home_directory_button)

        # SAVE DIRECTORY
        self.save_dir_label = QLabel("Save directory:")
        paths_layout.addWidget(self.save_dir_label)
        
        self.save_dir_selected_label = QLabel()
        paths_layout.addWidget(self.save_dir_selected_label)
        
        self.select_save_directory_button = QPushButton('Select Directory')
        self.select_save_directory_button.clicked.connect(lambda: self.open_directory_dialog(self.save_dir_selected_label))
        paths_layout.addWidget(self.select_save_directory_button)

        # CHECKPOINT DIRECTORY
        self.checkpoint_dir_label = QLabel("Checkpoint directory:")
        paths_layout.addWidget(self.checkpoint_dir_label)
        
        self.checkpoint_dir_selected_label = QLabel()
        paths_layout.addWidget(self.checkpoint_dir_selected_label)
        
        self.select_checkpoint_directory_button = QPushButton('Select Directory')
        self.select_checkpoint_directory_button.clicked.connect(lambda: self.open_directory_dialog(self.checkpoint_dir_selected_label))
        paths_layout.addWidget(self.select_checkpoint_directory_button)

        self.main_layout.addWidget(self.paths_groupbox, 0, 1)



        ## MUTATIONS ##
        self.mutations_groupbox = QGroupBox('Mutations')
        mutations_layout = QVBoxLayout()
        self.mutations_groupbox.setLayout(mutations_layout)

        # MUTATION TYPES
        self.list_view = QListView()
        self.list_view.setModel(self.generate_ticked_list())
        mutations_layout.addWidget(self.list_view)

        # L_M
        self.l_m_label = QLabel('l_m')
        mutations_layout.addWidget(self.l_m_label)
        self.l_m_edit = QLineEdit()
        mutations_layout.addWidget(self.l_m_edit)

        self.main_layout.addWidget(self.mutations_groupbox, 1, 0)



        ## GENOME ##
        self.genome_groupbox = QGroupBox('Genome')
        genome_layout = QVBoxLayout()
        self.genome_groupbox.setLayout(genome_layout)

        # G
        self.g_label = QLabel('g')
        genome_layout.addWidget(self.g_label)
        self.g_edit = QLineEdit()
        genome_layout.addWidget(self.g_edit)

        # Z_C
        self.z_c_label = QLabel('z_c')
        genome_layout.addWidget(self.z_c_label)
        self.z_c_edit = QLineEdit()
        genome_layout.addWidget(self.z_c_edit)

        # Z_NC
        self.z_nc_label = QLabel('z_nc')
        genome_layout.addWidget(self.z_nc_label)
        self.z_nc_edit = QLineEdit()
        genome_layout.addWidget(self.z_nc_edit)

        # HOMOGENEOUS
        self.homogeneous_checkbox = QCheckBox()
        self.homogeneous_checkbox.setText("Enable homogeneous genome")
        genome_layout.addWidget(self.homogeneous_checkbox)

        # ORIENTATION
        self.orientation_checkbox = QCheckBox()
        self.orientation_checkbox.setText("Enable one way genome")
        genome_layout.addWidget(self.orientation_checkbox)

        self.main_layout.addWidget(self.genome_groupbox, 1, 1)




        ## MUTATION RATES
        self.mutation_rates_groupbox = QGroupBox('Mutation rates')
        mutation_rates_layout = QVBoxLayout()
        self.mutation_rates_groupbox.setLayout(mutation_rates_layout)

        # POINT MUTATIONS RATE
        self.point_mutations_rate_label = QLabel('Point mutations rate')
        mutation_rates_layout.addWidget(self.point_mutations_rate_label)
        self.point_mutations_rate_edit = QLineEdit()
        mutation_rates_layout.addWidget(self.point_mutations_rate_edit)

        # SMALL INSERTIONS RATE
        self.small_insertions_rate_label = QLabel('Small insertions rate')
        mutation_rates_layout.addWidget(self.small_insertions_rate_label)
        self.small_insertions_rate_edit = QLineEdit()
        mutation_rates_layout.addWidget(self.small_insertions_rate_edit)

        # SMALL DELETIONS RATE
        self.small_deletions_rate_label = QLabel('Small deletions rate')
        mutation_rates_layout.addWidget(self.small_deletions_rate_label)
        self.small_deletions_rate_edit = QLineEdit()
        mutation_rates_layout.addWidget(self.small_deletions_rate_edit)

        # DELETIONS RATE
        self.deletions_rate_label = QLabel('Deletions rate')
        mutation_rates_layout.addWidget(self.deletions_rate_label)
        self.deletions_rate_edit = QLineEdit()
        mutation_rates_layout.addWidget(self.deletions_rate_edit)

        # DUPLICATIONS RATE
        self.duplications_rate_label = QLabel('Duplications rate')
        mutation_rates_layout.addWidget(self.duplications_rate_label)
        self.duplications_rate_edit = QLineEdit()
        mutation_rates_layout.addWidget(self.duplications_rate_edit)

        # INVERSIONS RATE
        self.inversions_rate_label = QLabel('Inversions rate')
        mutation_rates_layout.addWidget(self.inversions_rate_label)
        self.inversions_rate_edit = QLineEdit()
        mutation_rates_layout.addWidget(self.inversions_rate_edit)

        self.main_layout.addWidget(self.mutation_rates_groupbox, 2, 0)



        ## MUTAGENESE ##
        self.mutagenese_groupbox = QGroupBox('Mutagenese')
        mutagenese_layout = QVBoxLayout()
        self.mutagenese_groupbox.setLayout(mutagenese_layout)

        # ITERATIONS
        self.iterations_label = QLabel('Iterations')
        mutagenese_layout.addWidget(self.iterations_label)
        self.iterations_edit = QLineEdit()
        mutagenese_layout.addWidget(self.iterations_edit)

        self.main_layout.addWidget(self.mutagenese_groupbox, 2, 1)



        ## GENERATION BUTTON ##
        self.generate_button = QPushButton('Generate configuration file')
        self.generate_button.clicked.connect(self.generate_config)
        self.main_layout.addWidget(self.generate_button, 3, 0)


        ## SCROLL BAR ##
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.main_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)

    def generate_ticked_list(self):
        mutation_types = ["Point mutation", "Small insertion", "Small deletion", "Deletion", "Duplication", "Inversion"]
        model = QStandardItemModel()
        for mutation_type in mutation_types:
            item = QStandardItem(mutation_type)
            item.setCheckable(True)
            model.appendRow(item)
        return model
    

    def handleExperimentTypeChange(self):
        experiment_type = self.experiment_type_combo.currentText()
        if experiment_type == "Mutagenese":
            self.mutation_rates_groupbox.setEnabled(False)
            self.mutagenese_groupbox.setEnabled(True)
        else:
            self.mutation_rates_groupbox.setEnabled(True)
            self.mutagenese_groupbox.setEnabled(False)


    
    def generate_config(self):
        param1_value = self.param1_combo.currentText()
        param2_value = self.param2_edit.text()
        param3_items = [item.text() for item in self.param3_list.selectedItems()]
        param3_value = ', '.join(param3_items)

        print("Parameter 1:", param1_value)
        print("Parameter 2:", param2_value)
        print("Parameter 3:", param3_value)
    
    def open_directory_dialog(self, param):
        # Open a directory selection dialog
        directory_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory_path:
            param.setText(directory_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigGenerator()
    window.show()
    sys.exit(app.exec_())

