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

    def __init__(self, slab, surface_atoms=None, tol=1e-5, cutoff=5.0):
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

        self.sites_dict = self._get_higher_coordination_sites(
            coords_surf=self.coords_surf)
        
        self.coordinates = []
        self.r1_topology = []
        self.r2_topology = []
        self.connections = []
        self.names = []
        for name in self.sites_dict:
            coordinates, r1_topology, r2_topology = self.sites_dict[name]
            self.coordinates += coordinates
            self.r1_topology += r1_topology
            self.r2_topology += r2_topology
            self.connections += [self.connections_dict[name]]*len(coordinates)
            self.names += [name]*len(coordinates)
        self.coordinates = np.array(self.coordinates)
        self.r1_topology = np.array(self.r1_topology, dtype=object)
        self.r2_topology = np.array(self.r2_topology, dtype=object)
        self.connections = np.array(self.connections, dtype=int)
        self.frac_coords = np.dot(self.coordinates, np.linalg.pinv(slab.cell))
        self.names = np.array(self.names, dtype=object)
        self.ncoord_surf = self.get_coordination_numbers()
        self.n_sites = len(self.coordinates)
        
        self.centre = (np.sum(self.slab[surface_atoms].positions, axis=0) /
                       len(self.slab[surface_atoms]))[:2]*[0.95,0.90]

        self.screen = ((self.frac_coords[:, 0] > 0.-self.tol) &
                       (self.frac_coords[:, 0] < 1.-self.tol) &
                       (self.frac_coords[:, 1] > 0.-self.tol) &
                       (self.frac_coords[:, 1] < 1.-self.tol))

        self._symmetric_sites = None

    def get_coordination_numbers(self):
        """Return the coordination numbers of surface atoms."""
        
        ncoord = [sum(c) for c in self.slab.connectivity]
        ncoord *= self.repetitions
        ncoord_surf = np.array(ncoord)[self.index_surf]
        
        return ncoord_surf

    def get_number_of_connections(self, unique=True):
        """Return the number of connections associated with each site."""
        
        if unique:
            sel = self.get_symmetric_sites()
        else:
            sel = self.get_periodic_sites()

        return self.connections[sel]

    def get_coordinates(self, unique=True):
        """Return the 3D coordinates associated with each site."""
        
        if unique:
            sel = self.get_symmetric_sites()
        else:
            sel = self.get_periodic_sites()

        return self.coordinates[sel]

    def get_topology(self, unique=True):
        """Return the indices of adjacent surface atoms."""
        
        topology = [self.index_surf[t] for t in self.r1_topology]
        topology = np.array(topology, dtype=object)
        if unique:
            sel = self.get_symmetric_sites()
        else:
            sel = self.get_periodic_sites()

        return topology[sel]

    def get_names(self, unique=True):
        """Return the sites names."""
        
        if unique:
            sel = self.get_symmetric_sites()
        else:
            sel = self.get_periodic_sites()

        return self.names[sel]

    def _get_higher_coordination_sites(self,
                                       coords_surf,
                                       allow_obtuse=True):
        """Find all bridge and hollow sites (3-fold and 4-fold) given an
        input slab based Delaunay triangulation of surface atoms of a
        super-cell.

        TODO: Determine if this can be made more efficient by
        removing the 'sites' dictionary.

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
        
        periodic_sites = np.arange(self.frac_coords.shape[0])

        if screen:
            periodic_sites = periodic_sites[self.screen]

        else:
            periodic = periodic_sites.copy()[self.screen]
            for p in periodic:
                matched = utils.matching_sites(self.frac_coords[p],
                                               self.frac_coords)
                periodic_sites[matched] = p

        return periodic_sites

    def get_symmetric_sites(self,
                            unique=True,
                            screen=True,
                            sites_names=None,
                            topology_sym=False,
                            site_contains=None,
                            centre_in_cell=True):
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
        
        if self._symmetric_sites is None:
            sym = symmetry.Symmetry(self.slab, tol=self.tol)

            rotations, translations = sym.get_symmetry_operations(affine=False)
            rotations = np.swapaxes(rotations, 1, 2)

            affine = np.append(rotations, translations[:, None], axis=1)
            points = self.frac_coords
            true_index = self.get_periodic_sites(screen=False)

            affine_points = np.insert(points, 3, 1, axis=1)
            operations = np.dot(affine_points, affine)
            symmetric_sites = np.arange(points.shape[0])

            for i, j in enumerate(symmetric_sites):
                if i != j:
                    continue
                d = operations[i, :, None] - points
                d -= np.round(d)
                dind = np.where((np.abs(d) < self.tol).all(axis=2))[-1]
                symmetric_sites[np.unique(dind)] = true_index[i]

            if centre_in_cell is True:
                coords = self.coordinates
                for i, j in enumerate(symmetric_sites):
                    if (np.linalg.norm(coords[i][:2]-self.centre) <
                        np.linalg.norm(coords[j][:2]-self.centre)
                        and self.screen[i]):
                        indices = np.where(symmetric_sites == j)
                        symmetric_sites[indices] = i

            self._symmetric_sites = symmetric_sites

        if not topology_sym:
            symmetric_sites = self._symmetric_sites
            if screen:
                periodic_sites = self.get_periodic_sites(screen=screen)
                symmetric_sites = symmetric_sites[periodic_sites]
        
        else:
            symmetric_sites = self.get_periodic_sites(screen=screen)
            if centre_in_cell is True:
                coords = self.coordinates
                symmetric_sites = np.array(sorted(symmetric_sites, 
                    key=lambda n: np.linalg.norm(coords[n][:2]-self.centre)))
            site_tags = [self.get_site_tag(index=index) 
                         for index in symmetric_sites]
            _, indices = np.unique(site_tags, return_index=True)
            symmetric_sites = symmetric_sites[indices]
        
        if unique:
            symmetric_sites = np.unique(symmetric_sites)

        if site_contains is not None:
            mask = self._mask_site_contains(site_contains=site_contains,
                                            sites=symmetric_sites)
            symmetric_sites = symmetric_sites[mask]

        if sites_names is not None:
            mask = [self.names[s] in sites_names for s in symmetric_sites]
            symmetric_sites = symmetric_sites[mask]

        return symmetric_sites

    def get_adsorption_vectors(self, sites):
        """Returns the vectors representing the furthest distance from
        the neighboring atoms.

        Parameters
        ----------
        sites : numpy.ndarray (n,)
            Sites for which to calculate the adsorption vectors.

        Returns
        -------
        vectors : numpy.ndarray (n, 3)
            Adsorption vectors for surface sites.
        """
        
        coords = self.coordinates[sites]
        r1top = self.r1_topology[sites]
        r2top = self.r2_topology[sites]

        vectors = np.empty((coords.shape[0], 3))
        for i, _ in enumerate(coords):
            plane_points = np.array(np.hstack([r1top[i], r2top[i]]), dtype=int)
            vectors[i] = utils.plane_normal(self.coords_surf[plane_points])

        return vectors
    
    def get_adsorption_edges_linked(self,
                                    unique=True,
                                    screen=True,
                                    site_contains=None):
        """Return the edges of adsorption sites defined as all regions
        with adjacent vertices.

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

        site_id = self.get_symmetric_sites(unique=False, screen=False)
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
                                        unique=True,
                                        symmetric_ads=False,
                                        site_contains=None,
                                        centre_in_cell=True,
                                        **kwargs):
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

        sites_unique = self.get_symmetric_sites(**kwargs)
        sites_all = self.get_symmetric_sites(
            unique=False, screen=False, **kwargs)
        coords = self.coordinates[:, :2]
        r1_topology = self.r1_topology
        
        edges = []
        edges_sym = []
        uniques = []
        for s in sites_unique:
            diff = coords[:, None]-coords[s]
            norm = np.linalg.norm(diff, axis=2)
            neighbors = np.where((norm > range_edges[0]) &
                                 (norm < range_edges[1]))[0]
            neighbors_all = np.where((norm < range_edges[1]))[0]
            if centre_in_cell is True:
                neighbors = sorted(neighbors,
                     key=lambda n: np.linalg.norm(coords[n]-self.centre))
            for n in neighbors:
                if sites_names:
                    names = [names for names in sites_names
                             if [self.names[s], self.names[n]] == names[:2]]
                    if len(names) == 0:
                        continue
                    if len(sites_names) > 2:
                        conn = [m for m in neighbors_all if m != n
                                if self.names[m] in names[0][2]
                                if np.linalg.norm(coords[m]-coords[s])
                                   + np.linalg.norm(coords[m]-coords[n])
                                   - np.linalg.norm(coords[s]-coords[n])
                                   < self.tol*1e2]
                        if len(conn) == 0:
                            continue
                
                edge_new = [sites_all[s], sites_all[n],
                            np.round(norm[n,0], decimals=3)]
                dist_top = []
                for t in r1_topology[s]:
                    norm_i = np.linalg.norm(coords[n]-coords[t])
                    dist_top += [(sites_all[t], 
                                  np.round(norm_i, decimals=3))]
                edge_new += [sorted(dist_top)]
                if symmetric_ads is True:
                    edge_rev = [sites_all[n], sites_all[s],
                                np.round(norm[n,0], decimals=3)]
                    dist_top = []
                    for t in r1_topology[n]:
                        norm_i = np.linalg.norm(coords[s]-coords[t])
                        dist_top += [(sites_all[t],
                                      np.round(norm_i, decimals=3))]
                    edge_rev += [sorted(dist_top)]
                else:
                    edge_rev = edge_new
                if edge_new in edges_sym or edge_rev in edges_sym:
                    uniques += [False]
                else:
                    uniques += [True]
                    edges_sym += [edge_new]
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
        
        mask = [False]*len(sites)
        for i, site in enumerate(sites):
            if isinstance(site, (list, np.ndarray)):
                topologies = [i for e in site for i in self.r1_topology[e]]
            else:
                topologies = self.r1_topology[site]
            for t in topologies:
                symbol = self.slab[int(self.index_surf[t])].symbol
                if site_contains in (symbol, self.ncoord_surf[t]):
                    mask[i] = True

        return mask

    def plot(self, savefile=None):
        """Create a plot of the sites."""

        x_len = 1.2*(self.slab.cell[0][0]+self.slab.cell[1][0])
        y_len = 1.2*(self.slab.cell[0][1]+self.slab.cell[1][1])
        fig = plt.figure(figsize=(x_len, y_len), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])

        dt = scipy.spatial.Delaunay(
            self.coordinates[:, :2][self.connections == 1])
        ax.triplot(dt.points[:, 0], dt.points[:, 1], dt.simplices.copy(),
                   color='black')
        
        ax.plot(self.coordinates[:, 0][self.connections == 1],
                self.coordinates[:, 1][self.connections == 1],
                'o', color='r')
        ax.plot(self.coordinates[:, 0][self.connections == 2],
                self.coordinates[:, 1][self.connections == 2],
                'o', color='b')
        ax.plot(self.coordinates[:, 0][self.connections == 3],
                self.coordinates[:, 1][self.connections == 3],
                'o', color='y')
        ax.plot(self.coordinates[:, 0][self.connections == 4],
                self.coordinates[:, 1][self.connections == 4],
                'o', color='g')
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
        sites_sym = self.get_symmetric_sites()
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
                      **kwargs):
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
            sites = self.get_symmetric_sites(
                sites_names=sites_names,
                topology_sym=topology_sym,
                site_contains=site_contains)
            
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
                    **kwargs)]

        elif len(bonds) == 2:
            if linked_edges is True:
                edges = self.get_adsorption_edges_linked(
                    site_contains=site_contains)
            else:
                edges = self.get_adsorption_edges_not_linked(
                    sites_names=sites_names,
                    range_edges=range_edges,
                    symmetric_ads=symmetric_ads,
                    topology_sym=topology_sym,
                    site_contains=site_contains)
            
            if index in (None, -1):
                index = range(len(edges))
            elif isinstance(index, (int, np.integer)):
                index = [index]
            
            for i in index:
                slabs += [self._double_adsorption(
                    adsorbate=adsorbate,
                    edges=edges,
                    bonds=bonds,
                    edge_index=i,
                    auto_construct=auto_construct,
                    **kwargs)]

        else:
            raise ValueError('Only mono- and bidentate adsorption supported.')

        return slabs
    
    def add_adsorbate_ranges(self,
                             adsorbate,
                             centres,
                             slab=None,
                             bonds=None,
                             auto_construct=True,
                             ranges_coads=None,
                             sites_names=None,
                             site_contains=None,
                             intersection=True,
                             **kwargs):
        """Add an adsorbate to a slab at a desired distance ranges from the
        positions of centres.
        """
        
        if slab is None:
            slab = self.slab.copy()
        
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
        sites = np.unique(sites).tolist()
        
        if site_contains is not None:
            mask = self._mask_site_contains(site_contains=site_contains,
                                            sites=sites)
            sites = sites[mask]
        
        if sites_names is not None:
            mask = [self.names[s] in sites_names for s in sites]
            sites = sites[mask]
        
        slabs = []
        vectors = self.get_adsorption_vectors(sites=sites)
        for i, _ in enumerate(sites):
            slabs += [self._single_adsorption(
                adsorbate,
                bonds[0],
                sites=sites,
                vectors=vectors,
                slab=slab.copy(),
                site_index=i,
                auto_construct=auto_construct,
                **kwargs)]
        
        return slabs
        
    def coadsorption(self,
                     slabs_with_ads,
                     adsorbate,
                     bonds=None,
                     centres=None,
                     auto_construct=True,
                     ranges_coads=None,
                     sites_names=None,
                     site_contains=None,
                     **kwargs):
        """Add an adsorbate to a slab at a desired distance range from the
        position of centre.
        """
        
        slabs = []
        
        for slab in slabs_with_ads:
        
            if centres is None:
                centres = self._get_adsorbates_centres(slab=slab)

            slabs += self.add_adsorbate_ranges(
                adsorbate=adsorbate,
                bonds=bonds,
                centres=centres,
                slab=slab,
                auto_construct=auto_construct,
                ranges_coads=ranges_coads,
                sites_names=sites_names,
                site_contains=site_contains,
                **kwargs)

        return slabs

    def dissociation_reaction(self,
                              reactants,
                              products,
                              slab_reactants,
                              auto_construct=True,
                              ranges_coads=[0.1, 5.],
                              sites_names_list=None,
                              site_contains_list=None):
        """
        """
        slab_clean = self.slab.copy()
        
        if sites_names_list is None:
            sites_names_list = [None]*len(products)
        if site_contains_list is None:
            site_contains_list = [None]*len(products)
        
        centres_zero = self._get_adsorbates_centres(slab=slab_reactants)
        
        ranges_coads_new = [[0., ranges_coads[1]]]
        
        slabs_with_ads = self.add_adsorbate_ranges(
            adsorbate=products[0],
            centres=centres_zero,
            slab=slab_clean,
            auto_construct=auto_construct,
            ranges_coads=ranges_coads_new,
            sites_names=sites_names_list[0],
            site_contains=site_contains_list[0])
        
        slabs_products = []
        
        for slab in slabs_with_ads:
            
            centres = centres_zero+self._get_adsorbates_centres(slab=slab)
            ranges_coads_new += ranges_coads

            slabs_products += self.add_adsorbate_ranges(
                adsorbate=products[1],
                centres=centres,
                slab=slab,
                auto_construct=auto_construct,
                ranges_coads=ranges_coads,
                sites_names=sites_names_list[1],
                site_contains=site_contains_list[1])
        
        distances = []
        for slab_products in slabs_products:
            slab_products.positions[[17, 18]] = slab_products.positions[[18, 17]]
            
            distances += [self._get_distance_reactants_products(
                slab_reactants=slab_reactants,
                slab_products=slab_products,
            )]
    
        slabs_products = [x for _, x in sorted(zip(distances, slabs_products))]
    
        return slabs_products
    
    def _get_adsorbates_centres(self, slab):
        
        slab_clean = self.slab.copy()
        
        if len(slab) <= len(slab_clean):
            raise ValueError("No adsorbate present.")
        
        adsorbates_old = slab[len(slab_clean):]
        centres = [adsorbates_old.get_center_of_mass()]
    
        return centres
    
    def _get_distance_reactants_products(self, slab_reactants, slab_products,
                                         n_images=6):
        """
        """
        from ase.neb import NEB
        
        images = [slab_reactants]
        images += [slab_reactants.copy() for _ in range(n_images-2)]
        images += [slab_products]

        neb = NEB(images)
        neb.interpolate('idpp')
        distance = 0.
        for i, image in enumerate(images[1:]):
            diff = image.positions-images[i].positions
            norm = np.linalg.norm(diff, axis=1)
            distance += np.sum(norm)
        
        '''
        from ase.gui.gui import GUI
        gui = GUI(images)
        gui.run()
        '''
        
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
        site = sites[site_index]
        vector = self.get_adsorption_vectors(sites=[site])[0]
        if slab is None:
            slab = self.slab.copy()

        # Improved position estimate for site.
        u = self.r1_topology[site]
        r_site = radii[slab[self.index_surf[u]].numbers]
        r_bond = radii[atoms_ads.numbers[bond]]
        top_sites = self.coordinates[self.connections == 1]
        base_position = utils.trilaterate(top_sites[u], r_bond+r_site, vector)

        atoms_ads.translate(-atoms_ads.positions[bond])

        if auto_construct:
            atoms_ads = catkit.gen.molecules.get_3D_positions(atoms_ads, bond)

        if self.connections[site] == 2:
            coords_brg = self.coordinates[self.r1_topology[site]]
            direction = (coords_brg[1][:2]-coords_brg[0][:2])
            angle = np.arctan(direction[1]/(direction[0]+1e-10))
            atoms_ads.rotate(angle*180/np.pi, 'z')

        # Align with the adsorption vector
        atoms_ads.rotate([0, 0, 1], vector)

        atoms_ads.translate(base_position)
        n_atoms_slab = len(slab)
        slab += atoms_ads

        # Add graph connections
        for metal_index in self.index_surf[u]:
            slab.graph.add_edge(metal_index, bond+n_atoms_slab)

        # Get adsorption site tag
        tags = [self.get_site_tag(int(s)) for s in sites]
        number = len([tag for tag in tags[:site_index]
                      if tag == tags[site_index]])
        slab.site_tag = f'{tags[site_index]}-{number:02d}'

        return slab

    def _double_adsorption(self,
                           adsorbate,
                           bonds,
                           edges,
                           edge_index=0,
                           auto_construct=True,
                           slab=None):
        """Attach an adsorbate to two active sites."""
        
        atoms_ads = adsorbate.copy()
        graph_ads = atoms_ads.graph
        edge = edges[edge_index]
        coords_edge = self.coordinates[edge]
        if slab is None:
            slab = self.slab.copy()
        
        # Improved position estimate for site.
        r_bonds = radii[atoms_ads.numbers[bonds]] * 0.95
        topology_edge = self.r1_topology[edge]
        for i, u in enumerate(topology_edge):
            r_site = radii[slab[self.index_surf[u]].numbers] * 0.95
            top_sites = self.coordinates[self.connections == 1]
            coords_edge[i] = utils.trilaterate(top_sites[u], r_bonds[i]+r_site)

        v_site = coords_edge[1]-coords_edge[0]
        d_site = np.linalg.norm(v_site)
        uvec0 = v_site/d_site
        v_bond = atoms_ads[bonds[0]].position-atoms_ads[bonds[1]].position
        d_bond = np.linalg.norm(v_bond)
        dn = (d_bond-d_site)/2

        base_position0 = coords_edge[0]-uvec0*dn
        base_position1 = coords_edge[1]+uvec0*dn

        # Position the base atoms
        atoms_ads[bonds[0]].position = base_position0
        atoms_ads[bonds[1]].position = base_position1

        uvec1 = self.get_adsorption_vectors(sites=edge)
        uvec2 = np.cross(uvec1, uvec0)
        uvec2 /= -np.linalg.norm(uvec2, axis=1)[:, None]
        uvec1 = np.cross(uvec2, uvec0)

        center = adsorbate[bonds[0]].position
        center_new = atoms_ads[bonds[0]].position
        a1 = adsorbate[bonds[1]].position-center
        a2 = atoms_ads[bonds[1]].position-center_new
        b1 = [0, 0, 1]
        b2 = uvec1[0].tolist()
        rot_matrix = rotation_matrix(a1, a2, b1, b2)

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
                uvec = [-uvec0, uvec1[0], uvec2[0]]
                self._branch_bidentate(atoms_ads, uvec, branches0[0])
                for branch in branches0[1:]:
                    positions = catkit.gen.molecules._branch_molecule(
                        atoms_ads, branch, adsorption=True)
                    for j, b in enumerate(branch[1]):
                        atoms_ads[b].position = positions[j]
            
            branches1 = list(nx.bfs_successors(graph_ads, bonds[1]))
            if len(branches1[0][1]) != 0:
                uvec = [uvec0, uvec1[0], uvec2[0]]
                self._branch_bidentate(atoms_ads, uvec, branches1[0])
                for branch in branches1[1:]:
                    positions = catkit.gen.molecules._branch_molecule(
                        atoms_ads, branch, adsorption=True)
                    for j, b in enumerate(branch[1]):
                        atoms_ads[b].position = positions[j]
        
            for k in links:
                atoms_ads[k].position = np.dot(atoms_ads[k].position-center,
                                               rot_matrix.T)+center_new
                branches2 = list(nx.bfs_successors(graph_ads, k))
                for b in branches2[0][1]:
                    atoms_ads[b].position = np.dot(atoms_ads[b].position-center,
                                                   rot_matrix.T)+center_new

        else:
            other_atoms = [i for i, _ in enumerate(atoms_ads) 
                           if i not in bonds]

            for k in other_atoms:
                atoms_ads[k].position = np.dot(atoms_ads[k].position-center,
                                               rot_matrix.T)+center_new

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
        for i, u in enumerate(topology_edge):
            for metal_index in self.index_surf[u]:
                slab.graph.add_edge(metal_index, bonds[i]+n_atoms_slab)

        # Get adsorption site tag
        tags = ['-'.join([self.get_site_tag(int(e)) for e in edges[i]])
                for i in range(len(edges))]
        
        number = len([tag for tag in tags[:edge_index]
                      if tag == tags[edge_index]])
        
        slab.site_tag = f'{tags[edge_index]}-{number:02d}'

        return slab

    def _branch_bidentate(self, atoms, uvec, branch):
        """Return extended positions for additional adsorbates
        based on provided unit vectors.
        """
        
        r, nodes = branch
        num = atoms.numbers[[r] + nodes]
        d = radii[num[1:]] + radii[num[0]]
        c = atoms[r].position

        # Single additional atom
        if len(nodes) == 1:
            coord0 = c + \
                d[0] * uvec[0] * np.cos(1 / 3. * np.pi) + \
                d[0] * uvec[1] * np.sin(1 / 3. * np.pi)
            atoms[nodes[0]].position = coord0

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
            atoms[nodes[1]].position = coord1

        else:
            raise ValueError('Too many bonded atoms to position correctly.')

    def get_site_tag(self, index):
        """Return the tag of an active site."""
        
        site_tag = (self.names[index]+'['+
                    ','.join(sorted([f'{self.symbols[i]}.{self.ncoord_surf[i]}'
                              for i in self.r1_topology[index]]))+']')

        return site_tag
