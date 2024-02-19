import json
import sys

import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QListView, QMessageBox, QPushButton,
                             QScrollArea, QVBoxLayout, QWidget)


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
        experiment_layout = QGridLayout()
        self.experiment_groupbox.setLayout(experiment_layout)
        
        # EXPERIMENT NAME
        self.experiment_name_label = QLabel('Experiment name: ')
        experiment_layout.addWidget(self.experiment_name_label, 0, 0)
        self.experiment_name_edit = QLineEdit()
        experiment_layout.addWidget(self.experiment_name_edit, 0, 1)
        self.experiment_name_edit.textChanged.connect(self.handle_experiment_change)

        # EXPERIMENT TYPE
        self.experiment_type_label = QLabel('Experiment type: ')
        experiment_layout.addWidget(self.experiment_type_label, 1, 0)
        self.experiment_type_combo = QComboBox()
        self.experiment_type_combo.addItems(["Mutagenese", "Simulation"])
        experiment_layout.addWidget(self.experiment_type_combo, 1, 1)
        self.experiment_type_combo.currentIndexChanged.connect(self.handle_experiment_change)


        self.main_layout.addWidget(self.experiment_groupbox, 0, 0)



        ## PATHS ##
        self.paths_groupbox = QGroupBox('Paths')
        paths_layout = QGridLayout()
        self.paths_groupbox.setLayout(paths_layout)

        # HOME DIRECTORY
        self.home_dir_label = QLabel("Home directory: ")
        paths_layout.addWidget(self.home_dir_label, 0, 0)
        self.home_dir_selected_label = QLabel()
        paths_layout.addWidget(self.home_dir_selected_label, 0, 1)
        self.select_home_directory_button = QPushButton('Select Directory')
        self.select_home_directory_button.clicked.connect(lambda: self.open_directory_dialog(self.home_dir_selected_label))
        paths_layout.addWidget(self.select_home_directory_button, 0, 2)

        # SAVE DIRECTORY
        self.save_dir_label = QLabel("Save directory: ")
        paths_layout.addWidget(self.save_dir_label, 1, 0)
        self.save_dir_selected_label = QLabel()
        paths_layout.addWidget(self.save_dir_selected_label, 1, 1)
        self.select_save_directory_button = QPushButton('Select Directory')
        self.select_save_directory_button.clicked.connect(lambda: self.open_save_directory_dialog(self.save_dir_selected_label))
        paths_layout.addWidget(self.select_save_directory_button, 1, 2)

        # CHECKPOINT DIRECTORY
        self.checkpoint_dir_label = QLabel("Checkpoint directory: ")
        paths_layout.addWidget(self.checkpoint_dir_label, 2, 0)
        self.checkpoint_dir_selected_label = QLabel()
        paths_layout.addWidget(self.checkpoint_dir_selected_label, 2, 1)
        self.select_checkpoint_directory_button = QPushButton('Select Directory')
        self.select_checkpoint_directory_button.clicked.connect(lambda: self.open_directory_dialog(self.checkpoint_dir_selected_label))
        paths_layout.addWidget(self.select_checkpoint_directory_button, 2, 2)

        self.main_layout.addWidget(self.paths_groupbox, 0, 1)



        ## MUTATIONS ##
        self.mutations_groupbox = QGroupBox('Mutations')
        mutations_layout = QGridLayout()
        self.mutations_groupbox.setLayout(mutations_layout)

        # MUTATION TYPES
        self.mutation_type_list = QListView()
        self.mutation_type_model = self.generate_ticked_list()
        self.mutation_type_list.setModel(self.mutation_type_model)
        mutations_layout.addWidget(self.mutation_type_list, 0, 0, 1, 2)

        # L_M
        self.l_m_label = QLabel("l_m: ")
        mutations_layout.addWidget(self.l_m_label, 1, 0)
        self.l_m_edit = QLineEdit()
        self.l_m_edit.setText("10")
        mutations_layout.addWidget(self.l_m_edit, 1, 1)


        self.main_layout.addWidget(self.mutations_groupbox, 1, 0)



        ## GENOME ##
        self.genome_groupbox = QGroupBox('Genome')
        genome_layout = QGridLayout()
        self.genome_groupbox.setLayout(genome_layout)

        # G
        self.g_label = QLabel('g: ')
        genome_layout.addWidget(self.g_label, 0, 0)
        self.g_edit = QLineEdit()
        self.g_edit.setText("1e3")
        genome_layout.addWidget(self.g_edit, 0, 1, 1, 2)

        # AUTO_Z_C
        self.auto_z_c_checkbox = QCheckBox()
        self.auto_z_c_checkbox.setText("Keep z_c to: ")
        self.auto_z_c_checkbox.stateChanged.connect(self.auto_z_c_change)
        genome_layout.addWidget(self.auto_z_c_checkbox, 1, 0)

        self.z_c_factor_edit = QLineEdit()
        self.z_c_factor_edit.setText("1000")
        self.z_c_factor_edit.setEnabled(False)
        genome_layout.addWidget(self.z_c_factor_edit, 1, 1)

        self.z_c_factor_label = QLabel("g")
        genome_layout.addWidget(self.z_c_factor_label, 1, 2)
        
        # AUTO_Z_NC
        self.auto_z_nc_checkbox = QCheckBox()
        self.auto_z_nc_checkbox.setText("Keep z_nc to: ")
        self.auto_z_nc_checkbox.stateChanged.connect(self.auto_z_nc_change)
        genome_layout.addWidget(self.auto_z_nc_checkbox, 2, 0)

        self.z_nc_factor_edit = QLineEdit()
        self.z_nc_factor_edit.setText("1000")
        self.z_nc_factor_edit.setEnabled(False)
        genome_layout.addWidget(self.z_nc_factor_edit, 2, 1)

        self.z_nc_factor_label = QLabel("g")
        genome_layout.addWidget(self.z_nc_factor_label, 2, 2)

        # Z_C
        self.z_c_label = QLabel('z_c: ')
        genome_layout.addWidget(self.z_c_label, 3, 0)
        self.z_c_edit = QLineEdit()
        self.z_c_edit.setText("1e6")
        genome_layout.addWidget(self.z_c_edit, 3, 1, 1, 2)

        # Z_NC
        self.z_nc_label = QLabel('z_nc: ')
        genome_layout.addWidget(self.z_nc_label, 4, 0)
        self.z_nc_edit = QLineEdit()
        self.z_nc_edit.setText("1e6")
        genome_layout.addWidget(self.z_nc_edit, 4, 1, 1, 2)

        # HOMOGENEOUS
        self.homogeneous_checkbox = QCheckBox()
        self.homogeneous_checkbox.setText("Enable homogeneous genome")
        genome_layout.addWidget(self.homogeneous_checkbox, 5, 0)

        # ORIENTATION
        self.orientation_checkbox = QCheckBox()
        self.orientation_checkbox.setText("Enable one way genome")
        genome_layout.addWidget(self.orientation_checkbox, 5, 1)

        self.main_layout.addWidget(self.genome_groupbox, 1, 1, 1, 2)




        ## MUTATION RATES
        self.mutation_rates_groupbox = QGroupBox('Mutation rates')
        mutation_rates_layout = QGridLayout(self.mutation_rates_groupbox)
        self.mutation_rates_groupbox.setLayout(mutation_rates_layout)

        # POINT MUTATIONS RATE
        self.point_mutations_rate_label = QLabel('Point mutations rate: ')
        mutation_rates_layout.addWidget(self.point_mutations_rate_label, 0, 0)
        self.point_mutations_rate_edit = QLineEdit()
        self.point_mutations_rate_edit.setText('1e-9')
        mutation_rates_layout.addWidget(self.point_mutations_rate_edit, 0, 1)

        # SMALL INSERTIONS RATE
        self.small_insertions_rate_label = QLabel('Small insertions rate: ')
        mutation_rates_layout.addWidget(self.small_insertions_rate_label, 1, 0)
        self.small_insertions_rate_edit = QLineEdit()
        self.small_insertions_rate_edit.setText('1e-9')
        mutation_rates_layout.addWidget(self.small_insertions_rate_edit, 1, 1)

        # SMALL DELETIONS RATE
        self.small_deletions_rate_label = QLabel('Small deletions rate: ')
        mutation_rates_layout.addWidget(self.small_deletions_rate_label, 2, 0)
        self.small_deletions_rate_edit = QLineEdit()
        self.small_deletions_rate_edit.setText('1e-9')
        mutation_rates_layout.addWidget(self.small_deletions_rate_edit, 2, 1)

        # DELETIONS RATE
        self.deletions_rate_label = QLabel('Deletions rate: ')
        mutation_rates_layout.addWidget(self.deletions_rate_label, 3, 0)
        self.deletions_rate_edit = QLineEdit()
        self.deletions_rate_edit.setText('1e-9')
        mutation_rates_layout.addWidget(self.deletions_rate_edit, 3, 1)

        # DUPLICATIONS RATE
        self.duplications_rate_label = QLabel('Duplications rate: ')
        mutation_rates_layout.addWidget(self.duplications_rate_label, 4, 0)
        self.duplications_rate_edit = QLineEdit()
        self.duplications_rate_edit.setText('1e-9')
        mutation_rates_layout.addWidget(self.duplications_rate_edit, 4, 1)

        # INVERSIONS RATE
        self.inversions_rate_label = QLabel('Inversions rate: ')
        mutation_rates_layout.addWidget(self.inversions_rate_label, 5, 0)
        self.inversions_rate_edit = QLineEdit()
        self.inversions_rate_edit.setText('1e-9')
        mutation_rates_layout.addWidget(self.inversions_rate_edit, 5, 1)

        self.main_layout.addWidget(self.mutation_rates_groupbox, 2, 0)



        ## MUTAGENESE ##
        self.mutagenese_groupbox = QGroupBox('Mutagenese')
        mutagenese_layout = QGridLayout()
        self.mutagenese_groupbox.setLayout(mutagenese_layout)

        # ITERATIONS
        self.iterations_label = QLabel('Iterations: ')
        mutagenese_layout.addWidget(self.iterations_label, 0, 0)
        self.iterations_edit = QLineEdit()
        self.iterations_edit.setText("1e6")
        mutagenese_layout.addWidget(self.iterations_edit, 0, 1)

        # VARIABLE
        self.variable_label = QLabel("Variable: ")
        mutagenese_layout.addWidget(self.variable_label, 1, 0)
        self.variable_combo = QComboBox()
        self.variable_combo.addItems(["No variable", "g", "z_c", "z_nc"])
        mutagenese_layout.addWidget(self.variable_combo, 1, 1)
        self.variable_combo.currentIndexChanged.connect(self.variable_change)

        # RANGE
        self.range_widget = QWidget()
        range_layout = QGridLayout(self.range_widget)
        self.range_min_label = QLabel("From: ")
        range_layout.addWidget(self.range_min_label, 0, 0)
        self.range_min_edit = QLineEdit()
        self.range_min_edit.setText('1e2')
        range_layout.addWidget(self.range_min_edit, 0, 1)

        self.range_max_label = QLabel("To: ")
        range_layout.addWidget(self.range_max_label, 0, 2)
        self.range_max_edit = QLineEdit()
        self.range_max_edit.setText('1e6')
        range_layout.addWidget(self.range_max_edit, 0, 3)

        self.range_step_label = QLabel("\tPower step: ")
        range_layout.addWidget(self.range_step_label, 0, 4)
        self.range_step_edit = QLineEdit()
        self.range_step_edit.setText('1')
        range_layout.addWidget(self.range_step_edit, 0, 5)

        mutagenese_layout.addWidget(self.range_widget, 2, 0, 1, 2)

        self.main_layout.addWidget(self.mutagenese_groupbox, 2, 1)


        ## GENERATION BUTTON ##
        self.generate_button = QPushButton('Generate configuration file')
        self.generate_button.clicked.connect(self.generate_config)
        self.main_layout.addWidget(self.generate_button, 3, 0, 1, 2)


        ## SCROLL BAR ##
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.main_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)

        self.handle_experiment_change()

    def generate_ticked_list(self):
        mutation_types = ["Point mutation", "Small insertion", "Small deletion", "Deletion", "Duplication", "Inversion"]
        model = QStandardItemModel()
        for mutation_type in mutation_types:
            item = QStandardItem(mutation_type)
            item.setCheckable(True)
            model.appendRow(item)
        return model
    
    def variable_change(self):
        if self.variable_combo.currentText() != "No variable":
            self.range_widget.setEnabled(True)
        else:
            self.range_widget.setEnabled(False)
    
    def auto_z_c_change(self, state):
        if state == QtCore.Qt.Checked:
            self.z_c_edit.setEnabled(False)
            self.z_c_factor_edit.setEnabled(True)
        else:
            self.z_c_edit.setEnabled(True)
            self.z_c_factor_edit.setEnabled(False)
    
    def auto_z_nc_change(self, state):
        if state == QtCore.Qt.Checked:
            self.z_nc_edit.setEnabled(False)
            self.z_nc_factor_edit.setEnabled(True)
        else:
            self.z_nc_edit.setEnabled(True)
            self.z_nc_factor_edit.setEnabled(False)
    
    def handle_experiment_change(self):
        experiment_type = self.experiment_type_combo.currentText()
        experiment_name = self.experiment_name_edit.text()

        if experiment_name != "":
            for index in range(self.main_layout.count()):
                item = self.main_layout.itemAt(index)
                if isinstance(item.widget(), QWidget):
                    item.widget().setEnabled(True)
            if experiment_type == "Mutagenese":
                self.mutation_rates_groupbox.setEnabled(False)
            elif experiment_type == "Simulation":
                self.mutagenese_groupbox.setEnabled(False)
            if self.variable_combo.currentText() == "No variable":
                self.range_widget.setEnabled(False)
            if not self.auto_z_c_checkbox.isChecked:
                self.z_c_factor_edit.setEnabled(False)
            if not self.auto_z_nc_checkbox.isChecked:
                self.z_nc_factor_edit.setEnabled(False)
        else:
            for index in range(self.main_layout.count()):
                item = self.main_layout.itemAt(index)
                if isinstance(item.widget(), QWidget) and item.widget() != self.experiment_groupbox:
                    item.widget().setEnabled(False)
    
    def generate_config(self):
        d_params = {}

        ## EXPERIMENT
        d_params["Experiment"] = {
            "Experiment name": self.experiment_name_edit.text(),
            "Experiment type": self.experiment_type_combo.currentText(),
        }
        

        ## PATHS ##
        if self.home_dir_selected_label.text() == "":
            self.error_box("Please provide a home directory.")
            return None
        if self.save_dir_selected_label.text() == "":
            self.error_box("Please provide a save directory.")
            return None

        d_params["Paths"] = {
            "Home directory": self.home_dir_selected_label.text(),
            "Save directory": self.save_dir_selected_label.text(),
            "Checkpoint directory": self.checkpoint_dir_selected_label.text(),
        }

        ## MUTATIONS ##
        d_params["Mutations"] = {
            "Mutation types": [self.mutation_type_model.item(row).text() 
                               for row in range(self.mutation_type_model.rowCount()) 
                               if self.mutation_type_model.item(row).checkState() == QtCore.Qt.Checked],
            "l_m": self.l_m_edit.text(),
        }

        ## GENOME ##
        g = self.g_edit.text()
        d_params["Genome"] = {
            "g": g,
            "z_c": self.z_c_edit.text(),
            "z_c_auto": self.auto_z_c_checkbox.isChecked(),
            "z_c_factor": self.z_c_factor_edit.text(),
            "z_nc": self.z_nc_edit.text(),
            "z_nc_auto": self.auto_z_nc_checkbox.isChecked(),
            "z_nc_factor": self.z_nc_factor_edit.text(),
            "Homogeneous": self.homogeneous_checkbox.isChecked(),
            "Orientation": self.orientation_checkbox.isChecked(),
        }

        ## MUTATION RATES
        d_params["Mutation rates"] = {
            "Point mutation rate": self.point_mutations_rate_edit.text(),
            "Small insertion rate": self.small_insertions_rate_edit.text(),
            "Small deletion rate": self.small_deletions_rate_edit.text(),
            "Deletion rate": self.deletions_rate_edit.text(),
            "Duplication rate": self.duplications_rate_edit.text(),
            "Inversion rate": self.inversions_rate_edit.text(),
        }
        

        ## MUTAGENESE ##
        d_params["Mutagenese"] = {
            "Iterations": self.iterations_edit.text(),
            "Variable": self.variable_combo.currentText(),
            "From": self.range_min_edit.text(),
            "To": self.range_max_edit.text(),
            "Step": self.range_step_edit.text(),
        }

        ## SAVE FILE ##
        # options = QFileDialog.Options()
        # save_file, _ = QFileDialog.getSaveFileName(self, "Save file", "", "JSON files (*.json)", options=options)
        save_file = QFileDialog.getExistingDirectory(self, "Save file")
        if save_file:
            save_file += f"/{self.experiment_name_edit.text()}.json"
            with open(save_file, "w", encoding="utf8") as f:
                json.dump(d_params, f, indent=2)

                self.info_box("The configuration file was successfully generated at:\n"
                              f"{save_file}")

    def open_directory_dialog(self, param):
        directory_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory_path:
            param.setText(directory_path)
    
    def open_save_directory_dialog(self, param):
        directory_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory_path:
            param.setText(f"{directory_path}/{self.experiment_name_edit.text()}")
    
    def error_box(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(message)
        msgBox.setWindowTitle("Missing information")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

    def info_box(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(message)
        msgBox.setWindowTitle("Information")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigGenerator()
    window.show()
    sys.exit(app.exec_())

