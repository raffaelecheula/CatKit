from . import defaults
from . import utils
from . import symmetry
import catkit
import matplotlib.pyplot as plt
import itertools
import networkx as nx
import numpy as np
import scipy
from ase.build.tools import rotation_matrix

radii = defaults.get('radii')

class AdsorptionSites():
    """Adsorption site object."""

    def __init__(self, slab, surface_atoms=None, tol=1e-5, cutoff=5.0, 
                 centre_mult=(0.937, 0.895)):
        """Create an extended unit cell of the surface sites for
        use in identifying other sites.

        Parameters
        ----------
        slab : Gatoms object
            The slab associated with the adsorption site network to be
            attached.
        tol : float
            Absolute tolerance for floating point errors.
        """
        self.tol = tol
        self.slab = slab
        self.connections_dict = {'top': 1, 'brg': 2, '3fh': 3, '4fh': 4}

        if surface_atoms is None:
            surface_atoms = slab.get_surface_atoms()
        if surface_atoms is None:
            raise ValueError('Slab must contain surface atoms')
        
        index_all, coords_all = utils.expand_cell(slab, cutoff=cutoff)[:2]
        index_surf = np.where(np.in1d(index_all, surface_atoms))[0]
        self.repetitions = int(len(index_all)/len(slab))
        self.index_surf = index_all[index_surf]
        self.coords_surf = coords_all[index_surf]

        symbols_all = list(slab.symbols)*self.repetitions
        self.symbols = np.array(symbols_all)[self.index_surf]
        self.ncoord_surf = self.get_coordination_numbers()

        sites_dict = self._get_higher_coordination_sites(
            coords_surf=self.coords_surf)
        
        # Get sites.
        self.coordinates = []
        self.r1_topology = []
        self.r2_topology = []
        self.connections = []
        self.sites_names = []
        for name in sites_dict:
            coordinates, r1_topology, r2_topology = sites_dict[name]
            self.coordinates += coordinates
            self.r1_topology += r1_topology
            self.r2_topology += r2_topology
            self.connections += [self.connections_dict[name]]*len(coordinates)
            self.sites_names += [name]*len(coordinates)
        self.coordinates = np.array(self.coordinates)
        self.r1_topology = np.array(self.r1_topology, dtype=object)
        self.r2_topology = np.array(self.r2_topology, dtype=object)
        self.connections = np.array(self.connections, dtype=int)
        self.frac_coords = np.dot(self.coordinates, np.linalg.pinv(slab.cell))
        self.sites_names = np.array(self.sites_names, dtype=object)
        self.sites = np.arange(len(self.coordinates))
        self.n_sites = len(self.sites)
        
        # Order sites to have low indices close to the centre.
        self.centre_mult = centre_mult
        self.frac_centre = (
            np.sum(slab.get_scaled_positions(wrap=False)[surface_atoms],
            axis=0)/len(surface_atoms))[:2]*self.centre_mult
        self.centre = np.dot(self.frac_centre, slab.cell[:2,:2])
        self.indices = np.argsort(np.linalg.norm(
            self.frac_coords[:,:2]-self.frac_centre, axis=1))
        self.coordinates = self.coordinates[self.indices]
        self.r1_topology = self.r1_topology[self.indices]
        self.r2_topology = self.r2_topology[self.indices]
        self.connections = self.connections[self.indices]
        self.frac_coords = self.frac_coords[self.indices]
        self.sites_names = self.sites_names[self.indices]
        
        self.sites_to_surf = np.argsort(self.sites[self.indices])

        self.screen = ((self.frac_coords[:, 0] > 0.-self.tol) &
                       (self.frac_coords[:, 0] < 1.-self.tol) &
                       (self.frac_coords[:, 1] > 0.-self.tol) &
                       (self.frac_coords[:, 1] < 1.-self.tol))

        self.symmetric_sites = None
        self.topology_sym = None

    def get_coordination_numbers(self):
        """Return the coordination numbers of surface atoms."""
        
        ncoord = [sum(c) for c in self.slab.connectivity]
        ncoord *= self.repetitions
        ncoord_surf = np.array(ncoord)[self.index_surf]
        
        return ncoord_surf

    def get_number_of_connections(self, unique=True):
        """Return the number of connections associated with each site."""
        
        if unique:
            sel = self.get_adsorption_sites()
        else:
            sel = self.get_periodic_sites()

        return self.connections[sel]

    def get_coordinates(self, unique=True):
        """Return the 3D coordinates associated with each site."""
        
        if unique:
            sel = self.get_adsorption_sites()
        else:
            sel = self.get_periodic_sites()

        return self.coordinates[sel]

    def get_topology(self, unique=True):
        """Return the indices of adjacent surface atoms."""
        
        topology = [self.index_surf[t] for t in self.r1_topology]
        topology = np.array(topology, dtype=object)
        if unique:
            sel = self.get_adsorption_sites()
        else:
            sel = self.get_periodic_sites()

        return topology[sel]

    def get_names(self, unique=True):
        """Return the sites names."""
        
        if unique:
            sel = self.get_adsorption_sites()
        else:
            sel = self.get_periodic_sites()

        return self.sites_names[sel]

    def update_slab(self, slab_new, update_coordinates=False):
        """Update the slab."""
        
        if len(slab_new) != len(self.slab):
            raise ValueError(
                "Number of atoms of old and new slab must be equal!")

        self.slab.positions = slab_new.positions
        self.slab.symbols = slab_new.symbols
        self.slab.magmoms = slab_new.get_initial_magnetic_moments()
        symbols_all = list(self.slab.symbols)*self.repetitions
        self.symbols = np.array(symbols_all)[self.index_surf]

        if update_coordinates is True:
            self.coordinates = np.dot(self.frac_coords, slab_new.cell)
            self.coordinates -= (
                np.max(self.coords_surf)-np.max(slab_new.positions))
            self.centre = np.dot(self.frac_centre, slab_new.cell[:2,:2])

    def _get_higher_coordination_sites(self,
                                       coords_surf,
                                       allow_obtuse=True):
        """Find all bridge and hollow sites (3-fold and 4-fold) given an
        input slab based Delaunay triangulation of surface atoms of a
        super-cell.

        Parameters
        ----------
        coords_surf : numpy.ndarray (n, 3)
            Cartesian coordinates for the top atoms of the unit cell.

        Returns
        -------
        sites : dict of lists
            Dictionary sites containing positions, points, and neighbor lists.
        """
        
        sites = {
            'top': [[], [], []],
            'brg': [[], [], []],
            '3fh': [[], [], []],
            '4fh': [[], [], []],
        }
        sites['top'] = [coords_surf.tolist(),
                        [[i] for i in range(len(coords_surf))],
                        [[] for _ in coords_surf]]

        dt = scipy.spatial.Delaunay(coords_surf[:, :2])
        neighbors = dt.neighbors
        simplices = dt.simplices

        for i, corners in enumerate(simplices):
            cir = scipy.linalg.circulant(corners)
            edges = cir[:, 1:]

            # Inner angle of each triangle corner
            vec = coords_surf[edges.T]-coords_surf[corners]
            uvec = vec.T / np.linalg.norm(vec, axis=2).T
            angles = np.sum(uvec.T[0] * uvec.T[1], axis=1)

            # Angle types
            right = np.isclose(angles, 0)
            obtuse = (angles < -self.tol)

            rh_corner = corners[right]
            edge_neighbors = neighbors[i]

            if obtuse.any() and not allow_obtuse:
                # Assumption: All simplices with obtuse angles
                # are irrelevant boundaries.
                continue

            bridge = np.sum(coords_surf[edges], axis=1) / 2.0

            # Looping through corners allows for elimination of
            # redundant points, identification of 4-fold hollows,
            # and collection of bridge neighbors.
            for j, c in enumerate(corners):
                
                edge = sorted(edges[j])

                if edge in sites['brg'][1]:
                    continue

                # Get the bridge neighbors (for adsorption vector)
                neighbor_simplex = simplices[edge_neighbors[j]]
                oc = list(set(neighbor_simplex) - set(edge))[0]

                # Right angles potentially indicate 4-fold hollow
                potential_hollow = edge + sorted([c, oc])
                
                if c in rh_corner:

                    if potential_hollow in sites['4fh'][1]:
                        continue

                    # Assumption: If not 4-fold, this suggests
                    # no hollow OR bridge site is present.
                    ovec = coords_surf[edge] - coords_surf[oc]
                    ouvec = ovec / np.linalg.norm(ovec)
                    oangle = np.dot(*ouvec)
                    oright = np.isclose(oangle, 0)
                    if oright:
                        sites['4fh'][0] += [bridge[j]]
                        sites['4fh'][1] += [potential_hollow]
                        sites['4fh'][2] += [[]]
                        sites['top'][2][c] += [oc]
                
                else:
                    sites['brg'][0] += [bridge[j]]
                    sites['brg'][1] += [edge]
                    sites['brg'][2] += [[c, oc]]

                sites['top'][2][edge[0]] += [edge[1]]
                sites['top'][2][edge[1]] += [edge[0]]

            if not right.any() and not obtuse.any():
                hollow = np.average(coords_surf[corners], axis=0)
                sites['3fh'][0] += [hollow]
                sites['3fh'][1] += [corners.tolist()]
                sites['3fh'][2] += [[]]

        # For collecting missed bridge neighbors
        for s in sites['4fh'][1]:
        
            edges = itertools.product(s[:2], s[2:])
            for edge in edges:
                edge = sorted(edge)
                i = sites['brg'][1].index(edge)
                n, m = sites['brg'][1][i], sites['brg'][2][i]
                nn = list(set(s) - set(n + m))
        
                if len(nn) == 0:
                    continue
                sites['brg'][2][i] += [nn[0]]

        return sites

    def get_periodic_sites(self, screen=True):
        """Return the index of the sites which are unique by
        periodic boundary conditions.

        Parameters
        ----------
        screen : bool
            Return only sites inside the unit cell.

        Returns
        -------
        periodic_sites : numpy.ndarray (n,)
            Indices of the coordinates which are identical by
            periodic boundary conditions.
        """
        
        if screen:
            periodic_sites = self.sites[self.screen]

        else:
            periodic_sites = self.sites.copy()
            for p in self.sites[self.screen]:
                matched = utils.matching_sites(self.frac_coords[p],
                                               self.frac_coords)
                periodic_sites[matched] = p

        return periodic_sites

    def get_symmetric_sites(self,
                            topology_sym=False,
                            centre_in_cell=True):
        
        """Determine the symmetrically unique adsorption sites
        from a list of fractional coordinates.
        """
        
        if topology_sym is False:
            sym = symmetry.Symmetry(self.slab, tol=self.tol)

            rotations, translations = sym.get_symmetry_operations(affine=False)
            rotations = np.swapaxes(rotations, 1, 2)

            affine = np.append(rotations, translations[:, None], axis=1)
            points = self.frac_coords
            index = self.get_periodic_sites(screen=False)

            affine_points = np.insert(points, 3, 1, axis=1)
            operations = np.dot(affine_points, affine)
            symmetric_sites = np.arange(points.shape[0])

            for i, j in enumerate(symmetric_sites):
                if i != j:
                    continue
                d = operations[i, :, None] - points
                d -= np.round(d)
                dind = np.where((np.abs(d) < self.tol).all(axis=2))[-1]
                symmetric_sites[np.unique(dind)] = index[i]

            if centre_in_cell is True:
                coords = self.coordinates
                for i, j in enumerate(symmetric_sites):
                    if (np.linalg.norm(coords[i][:2]-self.centre) <
                        np.linalg.norm(coords[j][:2]-self.centre)
                        and self.screen[i]):
                        indices = np.where(symmetric_sites == j)
                        symmetric_sites[indices] = i

        else:
            symmetric_sites = self.sites.copy()
            if centre_in_cell is True:
                coords = self.coordinates
                symmetric_sites = np.array(sorted(symmetric_sites, 
                    key=lambda n: np.linalg.norm(coords[n][:2]-self.centre)))
            site_tags = [self.get_site_tag(index=index) 
                         for index in symmetric_sites]
            tags_unique, indices = np.unique(site_tags, return_index=True)
            tags_dict = dict(zip(tags_unique, indices))
            for i, t in enumerate(site_tags):
                index_sym = tags_dict[t]
                symmetric_sites[i] = symmetric_sites[index_sym]

        self.symmetric_sites = symmetric_sites
        self.topology_sym = topology_sym

        return symmetric_sites

    def get_adsorption_sites(self,
                             unique=True,
                             screen=True,
                             sites_names=None,
                             site_contains=None,
                             topology_sym=False,
                             centre_in_cell=True,
                             sites_avail=None):
        """Determine the symmetrically unique adsorption sites
        from a list of fractional coordinates.

        Parameters
        ----------
        unique : bool
            Return only the unique symmetrically reduced sites.
        screen : bool
            Return only sites inside the unit cell.
        sites_names : str
            Return only sites with given name.
        topology_sym : bool
            Calculate symmetry based on sites topology.

        Returns
        -------
        symmetric_sites : numpy.ndarray (n,)
            Array containing the indices of sites unique by symmetry.
        """
        
        if self.symmetric_sites is None or topology_sym != self.topology_sym:
            self.get_symmetric_sites(topology_sym=topology_sym,
                                     centre_in_cell=centre_in_cell)

        if sites_avail is None:
            sites = self.symmetric_sites.copy()
            periodic_sites = self.get_periodic_sites(screen=screen)
            sites = sites[periodic_sites]
        else:
            sites = sites_avail.copy()
        
        if unique:
            sites = np.unique(sites)

        if site_contains is not None:
            mask = self._mask_site_contains(site_contains=site_contains,
                                            sites=sites)
            sites = sites[mask]

        if sites_names is not None:
            mask = [self.sites_names[s] in sites_names for s in sites]
            sites = sites[mask]

        return sites

    def get_adsorption_vector(self, site):
        """Returns the vector representing the furthest distance from
        the neighboring atoms.

        Parameters
        ----------
        site : int
            Site for which to calculate the adsorption vector.

        Returns
        -------
        vectors : numpy.ndarray (3)
            Adsorption vector for surface site.
        """
        
        r1top = self.r1_topology[site]
        r2top = self.r2_topology[site]

        plane_points = np.array(np.hstack([r1top, r2top]), dtype=int)
        vector = utils.plane_normal(self.coords_surf[plane_points])

        return vector

    def get_adsorption_vector_edge(self, edge):
        """Returns the vectors representing the furthest distance from
        the neighboring atoms.

        Parameters
        ----------
        edge : numpy.ndarray (2)
            Edge for which to calculate the adsorption vector.

        Returns
        -------
        vectors : numpy.ndarray (3)
            Adsorption vector for edge.
        """
        
        r1top = self.r1_topology[edge]
        r2top = self.r2_topology[edge]

        plane_points = r1top[0]+r1top[1]
        if len(plane_points) < 3:
            plane_points += [i for i in r2top[0] if i in r2top[1]]
        if len(plane_points) < 3:
            plane_points += r2top[0]+r2top[1]
        
        vector = utils.plane_normal(self.coords_surf[plane_points])

        return vector

    def get_adsorption_edges_linked(self,
                                    sites_names=None,
                                    site_contains=None,
                                    symmetric_ads=False,
                                    topology_sym=False,
                                    sites_one=None,
                                    sites_two=None,
                                    unique=True,
                                    screen=True,
                                    centre_in_cell=True):
        """Return the edges of adsorption sites defined as all regions
        with adjacent vertices.

        TODO: sites_names, symmetric_ads, sites_one, sites_two,
        centre_in_cell

        Parameters
        ----------
        unique : bool
            Return only the symmetrically reduced edges.
        screen : bool
            Return edges which are unique via periodicity.

        Returns
        -------
        edges : numpy.ndarray (n, 2)
            All edges crossing ridge or vertices indexed by the expanded
            unit slab.
        """
        
        vt = scipy.spatial.Voronoi(self.coordinates[:, :2],
                                   qhull_options=f'Qbb Qc Qz C{1e-2}')

        select, lens = [], []
        for i, p in enumerate(vt.point_region):
            select += [vt.regions[p]]
            lens += [len(vt.regions[p])]

        dmax = max(lens)
        regions = np.zeros((len(select), dmax), int)
        mask = np.arange(dmax) < np.array(lens)[:, None]
        regions[mask] = np.concatenate(select)

        site_id = self.get_adsorption_sites(
            unique=False, screen=False, topology_sym=topology_sym)
        site_id = site_id + self.connections / 10.
        
        per = self.get_periodic_sites(screen=False)
        uper = self.get_periodic_sites(screen=True)
        
        edges, symmetry, uniques = [], [], []
        for i, p in enumerate(uper):
            poi = vt.point_region[p]
            voi = vt.regions[poi]

            for v in voi:
                nr = np.where(regions == v)[0]

                for n in nr:
                    edge = sorted((p, n))

                    if n in uper[:i + 1] or edge in edges:
                        continue

                    if np.in1d(per[edge], per[uper[:i]]).any() and screen:
                        continue

                    sym = sorted(site_id[edge])
                    if sym in symmetry:
                        uniques += [False]
                    else:
                        uniques += [True]
                        symmetry += [sym]

                    edges += [edge]

        edges = np.array(edges)
        if unique:
            edges = edges[uniques]

        if site_contains is not None:
            mask = self._mask_site_contains(site_contains=site_contains,
                                            sites=edges)
            edges = edges[mask]

        return edges
    
    def get_adsorption_edges_not_linked(self,
                                        range_edges=[0.1, 3.0],
                                        sites_names=None,
                                        site_contains=None,
                                        symmetric_ads=False,
                                        topology_sym=False,
                                        sites_one=None,
                                        sites_two=None,
                                        screen=True,
                                        unique=True,
                                        centre_in_cell=True):
        """Get bidentate adsorption sites, also the ones that are not
        directly connected.
        
        Parameters
        ----------
        range_edges : list (2)
            Minimum and maximum distance between sites.
        sites_names : list (n, 2)
            Define the site names allowed.
        unique : bool
            Return only the symmetrically reduced edges.
        symmetric_ads : bool
            Reduce the adsorption configurations if the adsorbate is symmetric.

        Returns
        -------
        edges : numpy.ndarray (n, 2)
            All edges crossing ridge or vertices indexed by the expanded
            unit slab.
        """

        if sites_one is None:
            if unique is True:
                sites_one = self.get_adsorption_sites(
                    topology_sym=topology_sym)
            elif screen is True:
                sites_one = self.get_periodic_sites()
            else:
                sites_one = self.sites
        
        sites_all = self.get_adsorption_sites(
            unique=False, screen=False, topology_sym=topology_sym)
        
        coords = self.coordinates[:, :2]
        r1top = self.r1_topology
        
        edges = []
        edges_sym = []
        uniques = []
        for s in sites_one:
            diff = coords[:, None]-coords[s]
            norm = np.linalg.norm(diff, axis=2)
            neighbors = np.where((norm > range_edges[0]) &
                                 (norm < range_edges[1]))[0]
            if sites_two is not None:
                neighbors = np.intersect1d(neighbors, sites_two)
            neighbors_all = np.where((norm < range_edges[1]))[0]
            if centre_in_cell is True:
                neighbors = sorted(neighbors,
                    key=lambda n: np.linalg.norm(coords[n]-self.centre))
            for n in neighbors:
                if sites_names:
                    names_list = [self.sites_names[s], self.sites_names[n]]
                    if isinstance(sites_names[0], dict):
                        names_matched = [names for names in sites_names
                                         if names_list == names['bonds']]
                        names_over = [name['over'] for name in names_matched]
                    else:
                        names_matched = [names for names in sites_names
                                         if names_list == names]
                    if len(names_matched) == 0:
                        continue
                    if (isinstance(sites_names[0], dict) 
                        and None not in names_over):
                        conn = [m for m in neighbors_all if m != n
                                if self.sites_names[m] in names_over
                                if np.linalg.norm(coords[m]-coords[s])
                                   + np.linalg.norm(coords[m]-coords[n])
                                   - np.linalg.norm(coords[s]-coords[n])
                                   < self.tol*1e2]
                        if len(conn) == 0:
                            continue
                # Vectors containing the unique index of the first site,
                # the unique index of the second site, their distance,
                # the distances to the surface atoms. They are used to screen
                # unique edges.
                edge_obj = [sites_all[s], sites_all[n],
                            np.round(norm[n,0], decimals=3)]
                dist_top = []
                for t in r1top[s]:
                    t = self.sites_to_surf[t]
                    norm_i = np.linalg.norm(coords[n]-coords[t])
                    dist_top += [(sites_all[t], 
                                  np.round(norm_i, decimals=3))]
                edge_obj += [sorted(dist_top)]
                if symmetric_ads is True:
                    edge_rev = [sites_all[n], sites_all[s],
                                np.round(norm[n,0], decimals=3)]
                    dist_top = []
                    for t in r1top[n]:
                        t = self.sites_to_surf[t]
                        norm_i = np.linalg.norm(coords[s]-coords[t])
                        dist_top += [(sites_all[t],
                                      np.round(norm_i, decimals=3))]
                    edge_rev += [sorted(dist_top)]
                else:
                    edge_rev = edge_obj
                if edge_obj in edges_sym or edge_rev in edges_sym:
                    uniques += [False]
                else:
                    uniques += [True]
                    edges_sym += [edge_obj]
                edges += [[s, n]]

        edges = np.array(edges)
        
        if unique is True:
            edges = edges[uniques]

        if site_contains is not None:
            mask = self._mask_site_contains(site_contains=site_contains,
                                            sites=edges)
            edges = edges[mask]
        
        return edges

    def _mask_site_contains(self, site_contains, sites):
        """Get the indices of the sites or edges that contain at least one
        atom with the element or the coordination number specified.
        """
        if not isinstance(site_contains, (list, np.ndarray)):
            site_contains = [site_contains]
        
        mask = [False]*len(sites)
        for i, site in enumerate(sites):
            if isinstance(site, (list, np.ndarray)):
                topologies = [j for e in site for j in self.r1_topology[e]]
            else:
                topologies = self.r1_topology[site]
            for t in topologies:
                symbol = self.slab[int(self.index_surf[t])].symbol
                ncoord = self.ncoord_surf[t]
                for c in site_contains:
                    if c in (symbol, ncoord):
                        mask[i] = True

        return mask

    def plot(self, scale_fig=1.2, scale_radii=0.90, delta_cell=0.50, 
             symbols_size=0.2, savefile=None):
        """Create a plot of the sites."""

        figsize = np.dot(self.slab.cell.T, np.ones(3))[:2]*scale_fig
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])

        from matplotlib import patches
        from ase.data import atomic_numbers, covalent_radii
        from ase.data.colors import jmol_colors

        screen_large = ((self.frac_coords[:, 0] > 0.-delta_cell-self.tol) &
                        (self.frac_coords[:, 0] < 1.+delta_cell+self.tol) &
                        (self.frac_coords[:, 1] > 0.-delta_cell-self.tol) &
                        (self.frac_coords[:, 1] < 1.+delta_cell+self.tol))

        for i in np.argsort(self.coords_surf[:, 2]):
            coords = self.coords_surf[i]
            if not screen_large[self.sites_to_surf[i]]:
                continue
            number = atomic_numbers[self.symbols[i]]
            radius = covalent_radii[number]*scale_radii
            color = jmol_colors[number]
            if not self.screen[self.sites_to_surf[i]]:
                color = (color+[1.,1.,1.])/2.
            circle = patches.Circle(xy=coords[:2], radius=radius, 
                                    facecolor=color, edgecolor='k')
            ax.add_patch(circle)

        cell_points = np.array([[0.,0.],
                                self.slab.cell[0][:2],
                                np.dot(self.slab.cell.T, np.ones(3))[:2],
                                self.slab.cell[1][:2],
                                [0.,0.]])
        ax.plot(*cell_points.T, color='k', linestyle='--', linewidth=1)

        obj_list = [
            [(self.connections == 1) & screen_large, (1.00, 0.50, 0.50)],
            [(self.connections == 2) & screen_large, (0.50, 0.75, 1.00)],
            [(self.connections == 3) & screen_large, (0.50, 0.90, 0.50)],
            [(self.connections == 4) & screen_large, (1.00, 1.00, 0.50)],
            [(self.connections == 1) & self.screen, (1.00, 0.00, 0.00)],
            [(self.connections == 2) & self.screen, (0.00, 0.50, 1.00)],
            [(self.connections == 3) & self.screen, (0.00, 0.80, 0.00)],
            [(self.connections == 4) & self.screen, (1.00, 1.00, 0.00)],
        ]
        
        for obj in obj_list:
            mask, color = obj
            coords = self.coordinates[mask][:,:2]
            for i, top in enumerate(self.r1_topology[mask]):
                if len(top) == 1:
                    continue
                dirs = self.coords_surf[top][:,:2]-coords[i]
                points = dirs/np.linalg.norm(dirs)*symbols_size+coords[i]
                coords = np.append(coords, points, axis=0)
            ax.plot(coords[:,0], coords[:,1], marker='o', markersize=12,
                    color=color, alpha=1.0, linestyle=' ')
            
            #for i, top in enumerate(self.r1_topology[mask]):
            #    if len(top) == 1:
            #        xy = self.coords_surf[top][0,:2]
            #        circle = patches.Circle(xy=xy, radius=symbols_size, 
            #                                facecolor=color)
            #        ax.add_patch(circle)
            #    else:
            #        dirs = self.coords_surf[top][:,:2]-coords[i]
            #        points = dirs/np.linalg.norm(dirs)*symbols_size+coords[i]
            #        #plt.plot(points[:,0], points[:,1], color=color, linewidth=10)
            #        patch = patches.PathPatch(points, facecolor=color)
            #        ax.add_patch(patch)

        ax.plot(*self.centre, marker='x', markersize=4, color='k')

        for i, coords in enumerate(self.coordinates):
            if screen_large[i]:
                ax.text(*coords[:2], i, horizontalalignment='center',
                        verticalalignment='center',)

        ax.axis('equal')
        ax.axis('off')
        if savefile:
            plt.savefig(savefile, transparent=True)
        else:
            plt.show()


class Builder(AdsorptionSites):
    """Initial module for creation of 3D slab structures with
    attached adsorbates.
    """

    def __repr__(self):
        formula = self.slab.get_chemical_formula()
        string = 'Adsorption builder for {} slab.\n'.format(formula)
        sites_sym = self.get_adsorption_sites()
        string += 'Unique adsorption sites: {}\n'.format(len(sites_sym))
        connections = self.get_number_of_connections()
        string += 'Sites number of connections: {}\n'.format(connections)
        edges = self.get_adsorption_edges_linked()
        string += 'Unique adsorption edges: {}'.format(len(edges))

        return string
    
    def add_adsorbate(self,
                      adsorbate,
                      bonds=None,
                      index=None,
                      auto_construct=True,
                      linked_edges=False,
                      range_edges=None,
                      sites_names=None,
                      symmetric_ads=False,
                      topology_sym=False,
                      site_contains=None,
                      slab=None,
                      screen=True,
                      unique=True,
                      sites_avail=None,
                      sites_list=None):
        """Add an adsorbate to a slab.

        Parameters
        ----------
        adsorbate : gratoms object
            Molecule to connect to the surface.
        bonds : int or list of 2 int
            Index of adsorbate atoms to be bonded.
        index : int
            Index of the site or edge to use as the adsorption position. A
            value of -1 will return all possible structures.
        auto_construct : bool
            Whether to automatically estimate the position of atoms in larger
            molecules or use the provided structure.
        linked_edges : bool
            Calculate edges only between directly connected sites.
        range_edges : list (2)
            Minimum and maximum distance between sites in bidentate adsorption.
        sites_names : list (n)
            Define the site names allowed.
        symmetric_ads : bool
            Reduce the adsorption configurations if the adsorbate is symmetric.
        topology_sym : bool
            Calculate symmetry based on sites topology.

        Returns
        -------
        slabs : gratoms object
            Slabs with adsorbate attached.
        """
        
        if bonds is None:
            bonds = np.where(adsorbate.get_tags() == -1)[0]

        slabs = []

        if len(bonds) == 0:
            raise ValueError('Specify the index of atoms to bond.')

        elif len(bonds) == 1:
            if sites_list is not None:
                sites = np.array(sites_list)
                if isinstance(sites[0], np.ndarray) and len(sites[0]) == 1:
                    sites = np.concatenate(sites)
            else:
                sites = self.get_adsorption_sites(
                    sites_names=sites_names,
                    site_contains=site_contains,
                    topology_sym=topology_sym,
                    screen=screen,
                    unique=unique,
                    sites_avail=sites_avail)
            
            if index in (None, -1):
                index = range(len(sites))
            elif isinstance(index, (int, np.integer)):
                index = [index]
            
            for i in index:
                slabs += [self._single_adsorption(
                    adsorbate=adsorbate,
                    bond=bonds[0],
                    sites=sites,
                    site_index=int(i),
                    auto_construct=auto_construct,
                    slab=slab)]

        elif len(bonds) == 2:
            if sites_list is not None:
                edges = np.array(sites_list)
            elif linked_edges is True:
                edges = self.get_adsorption_edges_linked(
                    sites_names=sites_names,
                    site_contains=site_contains,
                    symmetric_ads=symmetric_ads,
                    topology_sym=topology_sym,
                    screen=screen,
                    unique=unique,
                    sites_one=sites_avail,
                    sites_two=sites_avail)
            else:
                edges = self.get_adsorption_edges_not_linked(
                    range_edges=range_edges,
                    sites_names=sites_names,
                    site_contains=site_contains,
                    symmetric_ads=symmetric_ads,
                    topology_sym=topology_sym,
                    screen=screen,
                    unique=unique,
                    sites_one=sites_avail,
                    sites_two=sites_avail)
            
            if index in (None, -1):
                index = range(len(edges))
            elif isinstance(index, (int, np.integer)):
                index = [index]
            
            for i in index:
                slabs += [self._double_adsorption(
                    adsorbate=adsorbate,
                    bonds=bonds,
                    edges=edges,
                    edge_index=i,
                    auto_construct=auto_construct,
                    slab=slab)]

        else:
            raise ValueError('Only mono- and bidentate adsorption supported.')

        return slabs
    
    def add_adsorbate_ranges(self,
                             adsorbate,
                             centres,
                             slab=None,
                             bonds=None,
                             auto_construct=True,
                             range_edges=[0., 3.],
                             ranges_coads=None,
                             sites_names=None,
                             site_contains=None,
                             intersection=True,
                             topology_sym=False):
        """Add an adsorbate to a slab at a desired distance ranges from the
        positions of centres.
        """
        
        if bonds is None:
            bonds = np.where(adsorbate.get_tags() == -1)[0]
        
        if not isinstance(centres[0], (list, np.ndarray)):
            centres = [centres]
        
        if not isinstance(ranges_coads[0], (list, np.ndarray)):
            ranges_coads = [ranges_coads]*len(centres)
        
        coords = self.coordinates[:, :2]
        
        sites = []
        for i, centre in enumerate(centres):
            diff = coords[:, None]-centre[:2]
            norm = np.linalg.norm(diff, axis=2)
            sites += (np.where((norm > ranges_coads[i][0]) &
                               (norm < ranges_coads[i][1]))[0]).tolist()
        if intersection is True:
            sites = [i for i in sites if sites.count(i) == len(centres)]
        sites = np.unique(sites)
        
        slabs = self.add_adsorbate(
            adsorbate=adsorbate,
            bonds=bonds,
            auto_construct=auto_construct,
            range_edges=range_edges,
            linked_edges=False,
            sites_names=sites_names,
            symmetric_ads=False,
            topology_sym=False,
            screen=False,
            unique=False,
            site_contains=site_contains,
            slab=slab,
            sites_avail=sites,
        )
        
        return slabs
    
    def coadsorption(self,
                     slabs_with_ads,
                     adsorbate,
                     bonds=None,
                     centres=None,
                     auto_construct=True,
                     range_edges=[0., 3.],
                     ranges_coads=None,
                     sites_names=None,
                     site_contains=None):
        """Add an adsorbate to a slab at a desired distance range from the
        position of centre.
        """
        
        slabs = []
        
        for slab in slabs_with_ads:
        
            if centres is None:
                centres = self.get_adsorbates_centres(slab=slab)

            slabs += self.add_adsorbate_ranges(
                adsorbate=adsorbate,
                bonds=bonds,
                centres=centres,
                slab=slab,
                auto_construct=auto_construct,
                range_edges=range_edges,
                ranges_coads=ranges_coads,
                sites_names=sites_names,
                site_contains=site_contains)

        return slabs

    def dissociation_reaction(self,
                              reactant,
                              products,
                              bond_break,
                              bonds_surf,
                              slab_reactants,
                              auto_construct=True,
                              range_edges=[0., 3.],
                              displ_max_reaction=3.,
                              range_coadsorption=[0.1, 3.],
                              sites_names_list=None,
                              site_contains_list=None):
        """
        """
        
        slab_clean = self.slab.copy()
        n_atoms_clean = len(slab_clean)
        
        if sites_names_list is None:
            sites_names_list = [None]*2
        if site_contains_list is None:
            site_contains_list = [None]*2
        
        products, indices_frag = self.get_dissociation_fragments(
            reactant=reactant,
            bond_break=bond_break,
            bonds_surf=bonds_surf)
        
        indices_first = [f+n_atoms_clean for f in indices_frag[0]]
        ads_first = slab_reactants[indices_first]
        centres_first = [ads_first.get_center_of_mass()[:2]]
        ranges_fragments = [[0., displ_max_reaction]]
        
        indices_second = [f+n_atoms_clean for f in indices_frag[1]]
        ads_second = slab_reactants[indices_second]
        
        slabs_first_product = self.add_adsorbate_ranges(
            adsorbate=products[0],
            centres=centres_first,
            slab=slab_clean,
            auto_construct=auto_construct,
            range_edges=range_edges,
            ranges_coads=ranges_fragments,
            sites_names=sites_names_list[0],
            site_contains=site_contains_list[0])
        
        slabs_products = []
        
        for slab in slabs_first_product:
            
            centres_second = [ads_second.get_center_of_mass()[:2]]
            ranges_coads = ranges_fragments[:]
            for site in slab.info['site_numbers']:
                centres_second += [(np.sum(self.coordinates[site], axis=0) / 
                                    len(site))[:2]]
                ranges_coads += [range_coadsorption]

            slabs_products += self.add_adsorbate_ranges(
                adsorbate=products[1],
                centres=centres_second,
                slab=slab,
                auto_construct=auto_construct,
                range_edges=range_edges,
                ranges_coads=ranges_coads,
                sites_names=sites_names_list[1],
                site_contains=site_contains_list[1])
        
        distances = []
        indices = list(range(n_atoms_clean))
        indices += [j+n_atoms_clean for j in indices_frag[0]+indices_frag[1]]
        
        for i, _ in enumerate(slabs_products):
            slab = slabs_products[i][indices]
            
            distances += [self._get_distance_reactants_products(
                slab_reactants=slab_reactants[n_atoms_clean:],
                slab_products=slab[n_atoms_clean:])]
        
            site_numbers = slabs_products[i].info['site_numbers']
            slab.info['site_numbers'] = site_numbers
            slab.info['site_tag'] = '_'.join([self.get_site_tag(index=index) 
                                              for index in site_numbers])
            
            slabs_products[i] = slab
    
        indices_old = range(len(slabs_products))
        indices = [x for _, x in sorted(zip(distances, indices_old))]
        slabs_products = [slabs_products[i] for i in indices]
    
        return slabs_products
    
    def get_dissociation_fragments(self,
                                   reactant,
                                   bond_break,
                                   bonds_surf = []):
        
        reactant = reactant.copy()
        reactant.graph.remove_edge(*bond_break)
        
        fragments = []
        for bond_index in bond_break:
            succ = nx.bfs_successors(reactant.graph, bond_index)
            branches = [int(s) for list_s in succ for s in list_s[1]]
            fragment = [bond_index]+branches
            fragment.sort()
            fragments.append(fragment)
    
        tags = [-1 if i in bonds_surf else 0 for i, _ in enumerate(reactant)]
        reactant.set_tags(tags)
        
        products = [reactant.copy(), reactant.copy()]
        del products[0][fragments[1]]
        del products[1][fragments[0]]

        return products, fragments
    
    def get_adsorbates_centres(self, slab, slab_clean=None):
        
        if slab_clean is None:
            slab_clean = self.slab.copy()
        
        if len(slab) <= len(slab_clean):
            raise ValueError("No adsorbate present.")
        
        adsorbate = slab[len(slab_clean):]
        centres = [adsorbate.get_center_of_mass()]
    
        return centres
    
    def _get_distance_reactants_products(self, slab_reactants, slab_products,
                                         n_images=6):
        """
        """
        from ase.neb import NEB
        #from ase.neb import NEB, interpolate, idpp_interpolate
        #from ase.optimize.lbfgs import LBFGS
        
        images = [slab_reactants]
        images += [slab_reactants.copy() for _ in range(n_images-2)]
        images += [slab_products]

        neb = NEB(images)
        neb.interpolate('idpp')
        #steps_idpp = 1e3
        #fmax_idpp = 1e-2
        #interpolate(images=images, mic=False, apply_constraint=False)
        #idpp_interpolate(
        #    images    = images,
        #    traj      = None,
        #    log       = None,
        #    mic       = False,
        #    steps     = steps_idpp,
        #    fmax      = fmax_idpp,
        #    optimizer = LBFGS,
        #)

        distance = 0.
        for i, image in enumerate(images[1:]):
            diff = image.positions-images[i].positions
            norm = np.linalg.norm(diff, axis=1)
            distance += np.sum(norm)
        
        return distance
    
    def _single_adsorption(self,
                           adsorbate,
                           bond,
                           sites,
                           site_index=0,
                           auto_construct=True,
                           slab=None):
        """Attach an adsorbate to one active site."""
        
        atoms_ads = adsorbate.copy()
        site = int(sites[site_index])
        vector = self.get_adsorption_vector(site=site)
        if slab is None:
            slab = self.slab
        slab = slab.copy()

        # Improved position estimate for site.
        r1top = self.r1_topology[site]
        r_site = radii[slab[self.index_surf[r1top]].numbers]
        r_bond = radii[atoms_ads.numbers[bond]]
        base_position = utils.trilaterate(
            self.coords_surf[r1top], r_bond+r_site, vector)

        atoms_ads.translate(-atoms_ads.positions[bond])

        if auto_construct:
            atoms_ads = catkit.gen.molecules.get_3D_positions(atoms_ads, bond)

        if self.connections[site] == 2:
            coords_brg = self.coords_surf[r1top]
            direction = (coords_brg[1][:2]-coords_brg[0][:2])
            angle = np.arctan(direction[1]/(direction[0]+1e-10))
            atoms_ads.rotate(angle*180/np.pi, 'z')

        # Align with the adsorption vector
        atoms_ads.rotate([0, 0, 1], vector)

        atoms_ads.translate(base_position)
        n_atoms_slab = len(slab)
        slab += atoms_ads

        # Add graph connections
        for surf_index in self.index_surf[r1top]:
            slab.graph.add_edge(surf_index, bond+n_atoms_slab)

        # Store site numbers
        if 'site_numbers' in slab.info:
            slab.info['site_numbers'] += [[site]]
        else:
            slab.info['site_numbers'] = [[site]]

        # Get adsorption site tag
        tags = [self.get_site_tag(int(s)) for s in sites]
        tag_num = len([tag for tag in tags[:site_index]
                       if tag == tags[site_index]])
        slab.info['site_tag'] = tags[site_index]
        slab.info['tag_num'] = tag_num

        return slab

    def _double_adsorption(self,
                           adsorbate,
                           bonds,
                           edges,
                           edge_index=0,
                           auto_construct=False,
                           slab=None,
                           r_mult=0.95,
                           n_iter_max=10,
                           dn_thr=1e-5):
        """Attach an adsorbate to two active sites."""
        
        atoms_ads = adsorbate.copy()
        graph_ads = atoms_ads.graph
        edge = edges[edge_index]
        coords_edge = self.coordinates[edge]
        if slab is None:
            slab = self.slab
        slab = slab.copy()
        
        # Get adsorption vector
        zvectors = [self.get_adsorption_vector_edge(edge=edge)]*2

        old_positions = [
            atoms_ads[bonds[0]].position, atoms_ads[bonds[1]].position]
        new_positions = coords_edge.copy()

        # Iterate to position the adsorbate close to all the atoms of the edge
        for i in range(n_iter_max):
            r_bonds = radii[atoms_ads.numbers[bonds]]*r_mult
            for i, r1top in enumerate(self.r1_topology[edge]):
                r_site = radii[slab[self.index_surf[r1top]].numbers]*r_mult
                new_positions[i] = utils.trilaterate(
                    self.coords_surf[r1top], r_bonds[i]+r_site, 
                    zvector=zvectors[i])

            v_site = new_positions[1]-new_positions[0]
            d_site = np.linalg.norm(v_site)
            uvec0 = v_site/d_site
            v_bond = old_positions[1]-old_positions[0]
            d_bond = np.linalg.norm(v_bond)
            dn = (d_bond-d_site)/2

            new_positions = [
                new_positions[0]-uvec0*dn, new_positions[1]+uvec0*dn]

            for i in range(2):
                zvectors[i] = new_positions[i]-coords_edge[i]
                zvectors[i] /= np.linalg.norm(zvectors[i])
            
            if abs(dn) < dn_thr:
                break

        # Calculate the new adsorption vector, perpendicular to new uvec0.
        uvec2 = np.cross((zvectors[0]+zvectors[1])/2., uvec0)
        uvec1 = np.cross(uvec0, uvec2)

        a1 = old_positions[1]-old_positions[0]
        a2 = new_positions[1]-new_positions[0]
        b1 = [0, 0, 1]
        b2 = uvec1
        rot_matrix = rotation_matrix(a1, a2, b1, b2)

        # Rotate the adsorbate in the direction of the edge
        for k, _ in enumerate(atoms_ads):
            atoms_ads[k].position = np.dot(atoms_ads[k].position, rot_matrix.T)
        
        # Translate the adsorbate on the new (updated) edge positions
        atoms_ads.translate(new_positions[0]-old_positions[0])

        if auto_construct is True:
            # Temporarily break adsorbate bond
            if bonds in graph_ads.edges:
                links = []
                graph_ads.remove_edge(*bonds)
            else:
                links = [i for i in graph_ads.neighbors(bonds[0])
                         if i in graph_ads.neighbors(bonds[1])]
                for k in links:
                    graph_ads.remove_edge(bonds[0], k)
                    graph_ads.remove_edge(bonds[1], k)

            branches0 = list(nx.bfs_successors(graph_ads, bonds[0]))
            if len(branches0[0][1]) != 0:
                uvec = [-uvec0, uvec1, uvec2]
                self._branch_bidentate(atoms_ads, uvec, branches0[0])
                for branch in branches0[1:]:
                    positions = catkit.gen.molecules._branch_molecule(
                        atoms_ads, branch, adsorption=True)
                    for j, b in enumerate(branch[1]):
                        atoms_ads[b].position = positions[j]
            
            branches1 = list(nx.bfs_successors(graph_ads, bonds[1]))
            if len(branches1[0][1]) != 0:
                uvec = [uvec0, uvec1, uvec2]
                self._branch_bidentate(atoms_ads, uvec, branches1[0])
                for branch in branches1[1:]:
                    positions = catkit.gen.molecules._branch_molecule(
                        atoms_ads, branch, adsorption=True)
                    for j, b in enumerate(branch[1]):
                        atoms_ads[b].position = positions[j]
        
        n_atoms_slab = len(slab)
        slab += atoms_ads
        
        # Add graph connections
        if auto_construct is True:
            if links == []:
                slab.graph.add_edge(*np.array(bonds)+n_atoms_slab)
            else:
                for k in links:
                    slab.graph.add_edge(*np.array([bonds[0], k])+n_atoms_slab)
                    slab.graph.add_edge(*np.array([bonds[1], k])+n_atoms_slab)
        for i, r1top in enumerate(self.r1_topology[edge]):
            for surf_index in self.index_surf[r1top]:
                slab.graph.add_edge(surf_index, bonds[i]+n_atoms_slab)

        # Store site numbers
        if 'site_numbers' in slab.info:
            slab.info['site_numbers'] += [edge.tolist()]
        else:
            slab.info['site_numbers'] = [edge.tolist()]

        # Get adsorption site tag
        tags = [self.get_site_tag(e) for e in edges]
        tag_num = len([tag for tag in tags[:edge_index]
                       if tag == tags[edge_index]])
        slab.info['site_tag'] = tags[edge_index]
        slab.info['tag_num'] = tag_num

        return slab

    def _branch_bidentate(self, atoms, uvec, branch):
        """Return extended positions for additional adsorbates
        based on provided unit vectors.
        """
        
        r, nodes = branch
        num = atoms.numbers[[r] + nodes]
        d = radii[num[1:]] + radii[num[0]]
        c = atoms[r].position
        positions = atoms.positions

        # Single additional atom
        if len(nodes) == 1:
            coord0 = c + \
                d[0] * uvec[0] * np.cos(1 / 3. * np.pi) + \
                d[0] * uvec[1] * np.sin(1 / 3. * np.pi)
            positions[nodes[0]] = coord0
            atoms.positions = positions

        # Two branch system
        elif len(nodes) == 2:
            coord0 = c + \
                d[0] * uvec[1] * np.cos(1 / 3. * np.pi) + \
                0.866 * d[0] * uvec[0] * np.cos(1 / 3. * np.pi) + \
                0.866 * d[0] * uvec[2] * np.sin(1 / 3. * np.pi)
            atoms[nodes[0]].position = coord0
            coord1 = c + \
                d[1] * uvec[1] * np.cos(1 / 3. * np.pi) + \
                0.866 * d[1] * uvec[0] * np.cos(1 / 3. * np.pi) + \
                0.866 * d[1] * -uvec[2] * np.sin(1 / 3. * np.pi)
            positions[nodes[1]] = coord0
            atoms.positions = positions

        else:
            raise ValueError('Too many bonded atoms to position correctly.')

    def get_site_tag(self, index):
        """Return the tag of an active site."""
        
        if isinstance(index, (list, np.ndarray)):
            site_tag = '-'.join([self.get_site_tag(int(i)) for i in index])
        else:
            site_tag = (self.sites_names[index]+'['+','.join(sorted(
                [f'{self.symbols[r1top]}.{self.ncoord_surf[r1top]}'
                    for r1top in self.r1_topology[index]]))+']')
        
        return site_tag
