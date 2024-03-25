import json
from pathlib import Path
import sys

import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListView,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from utils import str_to_int, EPSILON


class ConfigGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Configuration File Generator")
        self.showMaximized()

        self.main_widget = QWidget()  # Create a widget to hold your layout
        self.main_layout = QGridLayout(self.main_widget)

        ## EXPERIMENT ##
        self.experiment_groupbox = QGroupBox("Experiment")
        experiment_layout = QGridLayout()
        self.experiment_groupbox.setLayout(experiment_layout)

        # EXPERIMENT NAME
        self.experiment_name_label = QLabel("Experiment name: ")
        experiment_layout.addWidget(self.experiment_name_label, 0, 0)
        self.experiment_name_edit = QLineEdit()
        experiment_layout.addWidget(self.experiment_name_edit, 0, 1)
        self.experiment_name_edit.textChanged.connect(self.handle_experiment_change)

        # EXPERIMENT TYPE
        self.experiment_type_label = QLabel("Experiment type: ")
        experiment_layout.addWidget(self.experiment_type_label, 1, 0)
        self.experiment_type_combo = QComboBox()
        self.experiment_type_combo.addItems(["Mutagenese", "Simulation"])
        experiment_layout.addWidget(self.experiment_type_combo, 1, 1)
        self.experiment_type_combo.currentIndexChanged.connect(
            self.handle_experiment_change
        )

        self.main_layout.addWidget(self.experiment_groupbox, 0, 0)

        ## PATHS ##
        self.paths_groupbox = QGroupBox("Paths")
        paths_layout = QGridLayout()
        self.paths_groupbox.setLayout(paths_layout)

        # SAVE DIRECTORY
        self.save_dir_label = QLabel("Save directory: ")
        paths_layout.addWidget(self.save_dir_label, 0, 0)
        self.save_dir_selected_label = QLabel()
        paths_layout.addWidget(self.save_dir_selected_label, 0, 1, 1, 3)
        self.select_save_directory_button = QPushButton("Select Directory")
        self.select_save_directory_button.clicked.connect(
            lambda: self.open_save_directory_dialog(self.save_dir_selected_label)
        )
        paths_layout.addWidget(self.select_save_directory_button, 0, 4)

        # CHECKPOINT BOX
        self.checkpoint_label = QLabel("Enable checkpointing: ")
        paths_layout.addWidget(self.checkpoint_label, 1, 0)
        self.checkpoint_checkbox = QCheckBox()
        paths_layout.addWidget(self.checkpoint_checkbox, 1, 1)
        self.checkpoint_checkbox.stateChanged.connect(self.handle_checkpoint_change)

        # CHECKPOINT NUMBER
        self.checkpoint_number_label = QLabel("Checkpoint number: ")
        paths_layout.addWidget(self.checkpoint_number_label, 1, 2)
        self.checkpoint_number_edit = QLineEdit()
        self.checkpoint_number_edit.setText("100")
        self.checkpoint_number_edit.textChanged.connect(self.handle_checkpoint_change)
        paths_layout.addWidget(self.checkpoint_number_edit, 1, 3)
        self.checkpoint_number_label_suffix = QLabel(f"")
        paths_layout.addWidget(self.checkpoint_number_label_suffix, 1, 4)

        # CHECKPOINT DIRECTORY
        self.checkpoint_dir_label = QLabel("Checkpoint directory: ")
        paths_layout.addWidget(self.checkpoint_dir_label, 2, 0)
        self.checkpoint_dir_selected_label = QLabel()
        paths_layout.addWidget(self.checkpoint_dir_selected_label, 2, 1, 1, 3)
        self.select_checkpoint_directory_button = QPushButton("Select Directory")
        self.select_checkpoint_directory_button.clicked.connect(
            lambda: self.open_directory_dialog(self.checkpoint_dir_selected_label)
        )
        paths_layout.addWidget(self.select_checkpoint_directory_button, 2, 4)

        self.main_layout.addWidget(self.paths_groupbox, 0, 1)

        ## MUTATIONS ##
        self.mutations_groupbox = QGroupBox("Mutations")
        mutations_layout = QGridLayout()
        self.mutations_groupbox.setLayout(mutations_layout)

        # MUTATION TYPES
        self.mutation_type_list = QListView()
        self.mutation_type_model = self.generate_ticked_list()
        self.mutation_type_list.setModel(self.mutation_type_model)
        self.mutation_type_model.itemChanged.connect(
            self.handle_mutation_selection_change
        )
        mutations_layout.addWidget(self.mutation_type_list, 0, 0, 1, 2)

        # L_M
        self.l_m_label = QLabel("l_m: ")
        mutations_layout.addWidget(self.l_m_label, 1, 0)
        self.l_m_edit = QLineEdit()
        self.l_m_edit.setText("10")
        mutations_layout.addWidget(self.l_m_edit, 1, 1)

        self.main_layout.addWidget(self.mutations_groupbox, 1, 0)

        ## GENOME ##
        self.genome_groupbox = QGroupBox("Genome")
        genome_layout = QGridLayout()
        self.genome_groupbox.setLayout(genome_layout)

        # G
        self.g_label = QLabel("g: ")
        genome_layout.addWidget(self.g_label, 0, 0)
        self.g_edit = QLineEdit()
        self.g_edit.setText("1e3")
        genome_layout.addWidget(self.g_edit, 0, 1, 1, 2)

        # AUTO_Z_C
        self.auto_z_c_checkbox = QCheckBox()
        self.auto_z_c_checkbox.setText("\u03B2: ")
        self.auto_z_c_checkbox.stateChanged.connect(self.auto_z_c_change)
        genome_layout.addWidget(self.auto_z_c_checkbox, 1, 0)

        self.z_c_factor_edit = QLineEdit()
        self.z_c_factor_edit.setText("1000")
        self.z_c_factor_edit.setEnabled(False)
        genome_layout.addWidget(self.z_c_factor_edit, 1, 1)

        # AUTO_Z_NC
        self.auto_z_nc_checkbox = QCheckBox()
        self.auto_z_nc_checkbox.setText("Initial \u03B1: ")
        self.auto_z_nc_checkbox.stateChanged.connect(self.auto_z_nc_change)
        genome_layout.addWidget(self.auto_z_nc_checkbox, 2, 0)

        self.z_nc_factor_edit = QLineEdit()
        self.z_nc_factor_edit.setText("1000")
        self.z_nc_factor_edit.setEnabled(False)
        genome_layout.addWidget(self.z_nc_factor_edit, 2, 1)

        # Z_C
        self.z_c_label = QLabel("z_c: ")
        genome_layout.addWidget(self.z_c_label, 3, 0)
        self.z_c_edit = QLineEdit()
        self.z_c_edit.setText("1e6")
        genome_layout.addWidget(self.z_c_edit, 3, 1, 1, 2)

        # Z_NC
        self.z_nc_label = QLabel("Initial z_nc: ")
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
        self.mutation_rates_groupbox = QGroupBox("Mutation rates")
        mutation_rates_layout = QGridLayout(self.mutation_rates_groupbox)
        self.mutation_rates_groupbox.setLayout(mutation_rates_layout)

        # POINT MUTATIONS RATE
        self.point_mutations_rate_label = QLabel("Point mutation rate: ")
        mutation_rates_layout.addWidget(self.point_mutations_rate_label, 0, 0)
        self.point_mutations_rate_edit = QLineEdit()
        self.point_mutations_rate_edit.setText("1e-9")
        mutation_rates_layout.addWidget(self.point_mutations_rate_edit, 0, 1)

        # SMALL INSERTIONS RATE
        self.small_insertions_rate_label = QLabel("Small insertion rate: ")
        mutation_rates_layout.addWidget(self.small_insertions_rate_label, 1, 0)
        self.small_insertions_rate_edit = QLineEdit()
        self.small_insertions_rate_edit.setText("1e-9")
        mutation_rates_layout.addWidget(self.small_insertions_rate_edit, 1, 1)

        # SMALL DELETIONS RATE
        self.small_deletions_rate_label = QLabel("Small deletion rate: ")
        mutation_rates_layout.addWidget(self.small_deletions_rate_label, 2, 0)
        self.small_deletions_rate_edit = QLineEdit()
        self.small_deletions_rate_edit.setText("1e-9")
        mutation_rates_layout.addWidget(self.small_deletions_rate_edit, 2, 1)

        # DELETIONS RATE
        self.deletions_rate_label = QLabel("Deletion rate: ")
        mutation_rates_layout.addWidget(self.deletions_rate_label, 3, 0)
        self.deletions_rate_edit = QLineEdit()
        self.deletions_rate_edit.setText("1e-9")
        mutation_rates_layout.addWidget(self.deletions_rate_edit, 3, 1)

        # DUPLICATIONS RATE
        self.duplications_rate_label = QLabel("Duplication rate: ")
        mutation_rates_layout.addWidget(self.duplications_rate_label, 4, 0)
        self.duplications_rate_edit = QLineEdit()
        self.duplications_rate_edit.setText("1e-9")
        mutation_rates_layout.addWidget(self.duplications_rate_edit, 4, 1)

        # INVERSIONS RATE
        self.inversions_rate_label = QLabel("Inversion rate: ")
        mutation_rates_layout.addWidget(self.inversions_rate_label, 5, 0)
        self.inversions_rate_edit = QLineEdit()
        self.inversions_rate_edit.setText("1e-9")
        mutation_rates_layout.addWidget(self.inversions_rate_edit, 5, 1)

        self.rate_edit_list = [
            self.point_mutations_rate_edit,
            self.small_insertions_rate_edit,
            self.small_deletions_rate_edit,
            self.deletions_rate_edit,
            self.duplications_rate_edit,
            self.inversions_rate_edit,
        ]

        self.main_layout.addWidget(self.mutation_rates_groupbox, 2, 0)

        ## MUTAGENESE ##
        self.mutagenese_groupbox = QGroupBox("Mutagenese")
        mutagenese_layout = QGridLayout()
        self.mutagenese_groupbox.setLayout(mutagenese_layout)

        # ITERATIONS
        self.iterations_label = QLabel("Iterations: ")
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
        self.range_min_edit.setText("1e2")
        range_layout.addWidget(self.range_min_edit, 0, 1)

        self.range_max_label = QLabel("To: ")
        range_layout.addWidget(self.range_max_label, 0, 2)
        self.range_max_edit = QLineEdit()
        self.range_max_edit.setText("1e6")
        range_layout.addWidget(self.range_max_edit, 0, 3)

        self.range_step_label = QLabel("\tPower step: ")
        range_layout.addWidget(self.range_step_label, 0, 4)
        self.range_step_edit = QLineEdit()
        self.range_step_edit.setText("1")
        range_layout.addWidget(self.range_step_edit, 0, 5)

        mutagenese_layout.addWidget(self.range_widget, 2, 0, 1, 2)

        self.main_layout.addWidget(self.mutagenese_groupbox, 2, 1)

        ## SIMULATION ##
        self.simulation_groupbox = QGroupBox("Simulation")
        simulation_layout = QGridLayout()
        self.simulation_groupbox.setLayout(simulation_layout)

        # REPLICATION MODEL #
        self.replication_model_label = QLabel("Replication model: ")
        simulation_layout.addWidget(self.replication_model_label, 0, 0)
        self.replication_model_combo = QComboBox()
        self.replication_model_combo.addItems(["Wright-Fisher", "Moran"])
        simulation_layout.addWidget(self.replication_model_combo, 0, 1)

        # GENERATION #
        self.generation_label = QLabel("Generation: ")
        simulation_layout.addWidget(self.generation_label, 1, 0)
        self.generation_edit = QLineEdit()
        self.generation_edit.setText("1e6")
        simulation_layout.addWidget(self.generation_edit, 1, 1)

        # POPULATION SIZE #
        self.population_size_label = QLabel("Population size: ")
        simulation_layout.addWidget(self.population_size_label, 2, 0)
        self.population_size_edit = QLineEdit()
        self.population_size_edit.setText("1e3")
        simulation_layout.addWidget(self.population_size_edit, 2, 1)

        # PLOT POINTS #
        self.plot_points_label = QLabel("Plot points: ")
        simulation_layout.addWidget(self.plot_points_label, 3, 0)
        self.plot_points_edit = QLineEdit()
        self.plot_points_edit.setText("10")
        simulation_layout.addWidget(self.plot_points_edit, 3, 1)

        self.main_layout.addWidget(self.simulation_groupbox, 3, 0)

        ## GENERATION BUTTON ##
        self.generate_button = QPushButton("Generate configuration file")
        self.generate_button.clicked.connect(self.generate_config)
        self.main_layout.addWidget(self.generate_button, 4, 0, 2, 2)

        ## SCROLL BAR ##
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.main_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)

        self.handle_experiment_change()

    def generate_ticked_list(self):
        mutation_types = [
            "Point mutation",
            "Small insertion",
            "Small deletion",
            "Deletion",
            "Duplication",
            "Inversion",
        ]
        model = QStandardItemModel()
        for mutation_type in mutation_types:
            item = QStandardItem(mutation_type)
            item.setCheckable(True)
            item.setCheckState(QtCore.Qt.Checked)
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
            self.handle_checkpoint_change()
            self.experiment_type_combo.setEnabled(True)
            for index in range(self.main_layout.count()):
                item = self.main_layout.itemAt(index)
                if isinstance(item.widget(), QWidget):
                    item.widget().setEnabled(True)

            if experiment_type == "Mutagenese":
                self.mutation_rates_groupbox.setEnabled(False)
                self.simulation_groupbox.setEnabled(False)
                self.save_dir_selected_label.setText(
                    str(Path(__file__).parent / "results/mutagenese" / experiment_name)
                )
            elif experiment_type == "Simulation":
                self.mutagenese_groupbox.setEnabled(False)
                self.simulation_groupbox.setEnabled(True)
                self.save_dir_selected_label.setText(
                    str(Path(__file__).parent / "results/simulation" / experiment_name)
                )
            if self.variable_combo.currentText() == "No variable":
                self.range_widget.setEnabled(False)
            if not self.auto_z_c_checkbox.isChecked:
                self.z_c_factor_edit.setEnabled(False)
            if not self.auto_z_nc_checkbox.isChecked:
                self.z_nc_factor_edit.setEnabled(False)
        else:
            self.experiment_type_combo.setEnabled(False)
            for index in range(self.main_layout.count()):
                item = self.main_layout.itemAt(index)
                if (
                    isinstance(item.widget(), QWidget)
                    and item.widget() != self.experiment_groupbox
                ):
                    item.widget().setEnabled(False)

    def generate_config(self):
        d_params = {}

        ## EXPERIMENT
        d_params["Experiment"] = {
            "Name": self.experiment_name_edit.text(),
            "Type": self.experiment_type_combo.currentText(),
        }

        ## PATHS ##
        if self.save_dir_selected_label.text() == "":
            self.error_box("Please provide a valid save directory.")
            return None
        checkpoint_number = 0
        if self.checkpoint_checkbox.isChecked():
            checkpoint_number = self.checkpoint_number_edit.text()
            if checkpoint_number == "":
                try:
                    checkpoint_number = str_to_int(checkpoint_number)
                except ValueError:
                    self.error_box(
                        f"Please provide a valid checkpoint number. (Yours is {checkpoint_number})"
                    )
                    return None
            if checkpoint_number == 0:
                self.info_box("The checkpoint number can't be 0. Defaulting to 1.")
                checkpoint_number = 1
            elif self.checkpoint_dir_selected_label.text() == "":
                self.error_box("Please provide a valid checkpoint directory.")
                return None

        d_params["Paths"] = {
            "Save": self.save_dir_selected_label.text(),
            "Checkpointing": self.checkpoint_checkbox.isChecked(),
            "Checkpoint number": checkpoint_number,
            "Checkpoint": self.checkpoint_dir_selected_label.text(),
        }

        ## MUTATIONS ##
        d_params["Mutations"] = {
            "Mutation types": [
                self.mutation_type_model.item(row).text()
                for row in range(self.mutation_type_model.rowCount())
                if self.mutation_type_model.item(row).checkState() == QtCore.Qt.Checked
            ],
            "l_m": self.l_m_edit.text(),
        }

        def var_is_ok(var):
            try:
                str_to_int(var)
                return True
            except ValueError:
                self.error_box(f"Please provide a valid value. (Yours is '{var}')")
                return False

        ## GENOME ##
        g = self.g_edit.text()
        if not var_is_ok(g):
            return None
        z_c = self.z_c_edit.text()
        if not var_is_ok(z_c):
            return None
        z_nc = self.z_nc_edit.text()
        if not var_is_ok(z_nc):
            return None
        z_c_factor = self.z_c_factor_edit.text()
        if not var_is_ok(z_c_factor):
            return None
        z_nc_factor = self.z_nc_factor_edit.text()
        if not var_is_ok(z_nc_factor):
            return None

        d_params["Genome"] = {
            "g": g,
            "z_c": z_c,
            "z_c_auto": self.auto_z_c_checkbox.isChecked(),
            "z_c_factor": z_c_factor,
            "z_nc": z_nc,
            "z_nc_auto": self.auto_z_nc_checkbox.isChecked(),
            "z_nc_factor": z_nc_factor,
            "Homogeneous": self.homogeneous_checkbox.isChecked(),
            "Orientation": self.orientation_checkbox.isChecked(),
        }

        ## MUTATION RATES
        point_mutations_rate = self.point_mutations_rate_edit.text()
        if not var_is_ok(point_mutations_rate):
            return None
        small_insertions_rate = self.small_insertions_rate_edit.text()
        if not var_is_ok(small_insertions_rate):
            return None
        small_deletions_rate = self.small_deletions_rate_edit.text()
        if not var_is_ok(small_deletions_rate):
            return None
        deletions_rate = self.deletions_rate_edit.text()
        if not var_is_ok(deletions_rate):
            return None
        duplications_rate = self.duplications_rate_edit.text()
        if not var_is_ok(duplications_rate):
            return None
        inversions_rate = self.inversions_rate_edit.text()
        if not var_is_ok(inversions_rate):
            return None

        d_params["Mutation rates"] = {
            "Point mutation": point_mutations_rate,
            "Small insertion": small_insertions_rate,
            "Small deletion": small_deletions_rate,
            "Deletion": deletions_rate,
            "Duplication": duplications_rate,
            "Inversion": inversions_rate,
        }
        ## MUTAGENESE ##
        iterations = self.iterations_edit.text()
        if not var_is_ok(iterations):
            return None
        range_min = self.range_min_edit.text()
        if not var_is_ok(range_min):
            return None
        range_max = self.range_max_edit.text()
        if not var_is_ok(range_max):
            return None
        range_step = self.range_step_edit.text()
        if not var_is_ok(range_step):
            return None

        d_params["Mutagenese"] = {
            "Iterations": iterations,
            "Variable": self.variable_combo.currentText(),
            "From": range_min,
            "To": range_max,
            "Step": range_step,
        }

        ## SIMULATION ##
        generation = self.generation_edit.text()
        if not var_is_ok(generation):
            return None
        population_size = self.population_size_edit.text()
        if not var_is_ok(population_size):
            return None
        plot_points = self.plot_points_edit.text()
        if not var_is_ok(plot_points):
            return None
        if str_to_int(plot_points) - str_to_int(generation) > EPSILON:
            self.error_box(
                "The number of plot points must be smaller than or equal to the number of generations."
            )
            return None
        if str_to_int(plot_points) == 0:
            self.info_box("The number of plot points can't be 0. Defaulting to 1.")
            plot_points = 1

        d_params["Simulation"] = {
            "Replication model": self.replication_model_combo.currentText(),
            "Generations": generation,
            "Population size": population_size,
            "Plot points": plot_points,
        }

        ## SAVE FILE ##
        # options = QFileDialog.Options()
        # save_file, _ = QFileDialog.getSaveFileName(self, "Save file", "", "JSON files (*.json)", options=options)
        save_file = QFileDialog.getExistingDirectory(self, "Save file")
        if save_file:
            save_file += f"/{self.experiment_name_edit.text()}.json"
            with open(save_file, "w", encoding="utf8") as f:
                json.dump(d_params, f, indent=2)

                self.info_box(
                    "The configuration file was successfully generated at:\n"
                    f"{save_file}"
                )

    def handle_checkpoint_change(self):
        if self.checkpoint_checkbox.isChecked():
            self.checkpoint_number_edit.setEnabled(True)
            self.checkpoint_dir_selected_label.setEnabled(True)
            self.checkpoint_dir_selected_label.setText(
                str(
                    Path(__file__).parent
                    / "checkpoint"
                    / self.experiment_type_combo.currentText().lower()
                    / self.experiment_name_edit.text()
                )
            )
            try:
                self.checkpoint_number_label_suffix.setText(
                    f"(Every {str_to_int(self.generation_edit.text()) // str_to_int(self.checkpoint_number_edit.text())} generations)"
                )
            except (ValueError, ZeroDivisionError):
                self.checkpoint_number_label_suffix.setText("")
        else:
            self.checkpoint_number_edit.setEnabled(False)
            self.checkpoint_dir_selected_label.setEnabled(False)
            self.checkpoint_number_label_suffix.setText("")

    def handle_mutation_selection_change(self):
        for edit in self.rate_edit_list:
            edit.setEnabled(False)
        for row in range(self.mutation_type_model.rowCount()):
            if self.mutation_type_model.item(row).checkState() == QtCore.Qt.Checked:
                self.rate_edit_list[row].setEnabled(True)
        if (
            self.mutation_type_model.item(1).checkState() == QtCore.Qt.Checked
            or self.mutation_type_model.item(2).checkState() == QtCore.Qt.Checked
        ):
            self.l_m_edit.setEnabled(True)
        else:
            self.l_m_edit.setEnabled(False)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigGenerator()
    window.show()
    sys.exit(app.exec_())
