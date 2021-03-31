import numpy as np
from functools import reduce

from so3g.proj import Ranges, RangesMatrix
from . import AxisManager

class FlagManager(AxisManager):
    """An extension of the AxisManager class to make functions 
    more specifically associated with cuts and flags.
    
    FlagManagers must have a dets axis and a samps axis when created.
    
    FlagManager only expects to have individual flags that are mapped to the 
    detector axis, the sample axis, or both. 
    
    Detector Flags can be passed as bitmasks or boolean arrays. To match with
    Ranges and RangesMatrix, the default is False and the exceptions
    are True
    """
    
    def __init__(self, det_axis, samp_axis):
        self._det_name = det_axis.name
        self._samp_name = samp_axis.name
        
        super().__init__(det_axis, samp_axis)
        
        ## are these checks required anymore?
        if not self._det_name in self._axes:
            raise ValueError('FlagManagers require a dets axis')
        if not self._samp_name in self._axes:
            raise ValueError('FlagManagers require a samps axis')
    
    def wrap(self, name, data, axis_map=None, **kwargs):
        """See core.AxisManager for basic usage
        
        If axis_map is None, the data better be (dets,), (samps,),
            or (dets, samps). Will not work if dets.count == samps.count
            
        """
        
        if axis_map is None:
            if self[self._det_name].count == self[self._samp_name].count:
                raise ValueError("Cannot auto-detect axis_map when dets and "
                                 "samps axes have equal lengths. axis_map "
                                 "must be defined")
            s = _get_shape(data)
            
            if len(s) == 1:
                if s[0] == self[self._det_name].count:
                    ## detector only flag. Turn into RangesMatrix
                    axis_map=[(0,self[self._det_name])]
                elif s[0] == self[self._samp_name].count:
                    axis_map=[(0, self.samps)]
                else:
                    raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
            elif len(s) == 2:
                if s[0] == self[self._det_name].count and s[1] == self[self._samp_name].count:
                    axis_map=[(0,self[self._det_name]), (1,self.samps)]
                elif s[1] == self[self._det_name].count and s[0] == self[self._samp_name].count:
                    raise ValueError("FlagManager only takes 2D data aligned as"
                                     " (dets, samps). Data of shape {}"
                                     " is the wrong shape".format(s))
                else:
                    raise ValueError("FlagManager only takes 2D data aligned as"
                                     " (dets, samps). Data of shape {}"
                                     " is the wrong shape".format(s))
            else:
                raise ValueError("FlagManager only takes data aligned with"
                                     " dets and/or samps. Data of shape {}"
                                     " is the wrong shape".format(s))
        
        if len(axis_map)==1 and axis_map[0][1]== self._det_name:
            ### Change detector flags to RangesMatrix in the backend
            x = Ranges(self.samps.count)
            data = RangesMatrix([Ranges.ones_like(x) if Y 
                                 else Ranges.zeros_like(x) for Y in data])
            axis_map = [(0,self[self._det_name]),(1,self[self._samp_name])]

        super().wrap(name, data, axis_map, **kwargs)

    def wrap_dets(self, name, data):
        """Adding flag with just (dets,) axis.
        """
        s = _get_shape(data)
        if not len(s) == 1 or s[0] != self[self._det_name].count:
            raise ValueError("Data of shape {} is cannot be aligned with"
                             "the detector axis".format(s))
        self.wrap(name, data, axis_map=[(0,self._det_name)])
        
    def wrap_samps(self, name, data):
        """Adding flag with just (samps,) axis.
        """
        s = _get_shape(data)
        if not len(s) == 1 or s[0] != self[self._samp_name].count:
            raise ValueError("Data of shape {} is cannot be aligned with"
                             "the samps axis".format(s))
        self.wrap(name, data, axis_map=[(0,self._samp_name)])
        
    def wrap_dets_samps(self, name, data):
        """Adding flag with (dets, samps) axes.
        """
        s = _get_shape(data)
        if (not len(s) == 2 or s[0] != self[self._det_name].count or
               s[1] != self[self._samp_name].count):
            raise ValueError("Data of shape {} is cannot be aligned with"
                             "the (dets,samps) axss".format(s))
        self.wrap(name, data, axis_map=[(0,self._det_name), (1,self._samp_name)])
        
    def copy(self, axes_only=False):
        out = FlagManager(self[self._det_name], self[self._samp_name])
        for k, v in self._axes.items():
            out._axes[k] = v
        if axes_only:
            return out
        for k, v in self._fields.items():
            out._fields[k] = v.copy()
        for k, v in self._assignments.items():
            out._assignments[k] = v.copy()
        return out
    
    def get_zeros(self):
        """
        Return a correctly sized RangesMatrix for building cuts for the FlagManager
        """
        return RangesMatrix([Ranges(self[self._samp_name].count) for det in self[self._det_name].vals])
        
    def buffer(self, n_buffer, flags=None):
        """Buffer all the samps cuts by n_buffer
        Like with Ranges / Ranges Matrix, buffer changes everything in place        
        
        Args:
            n_buffer: number of samples to buffer the samps cuts
            flags: List of flags to buffer. Uses their names
        """
        if flags is None:
            flags = self._fields
        
        for f in flags:
            self[f].buffer(n_buffer)
        
    def buffered(self, n_buffer, flags=None):
        """Return new FlagManager that has all the samps cuts buffered by n_buffer
        Like with Ranges / Ranges Matrix, buffered returns new object
        
        Args:
            n_buffer: number of samples to buffer the samps cuts
            flags: List of flags to buffer. Uses their names
        
        Returns:
            new: FlagManager with all flags buffered
        """
        new = self.copy()
        new.buffer(n_buffer, flags)
        return new
        
    def reduce(self, flags=None, method='union', wrap=False, new_flag=None,
               remove_reduced=False):
        """Reduce (combine) flags in the FlagManager together. 
        
        Args:
            flags: List of flags to collapse together. Uses their names.
                   If flags is None then all flags are reduced
            method: How to collapse the data. Accepts 'union','intersect',
                        or function.
            wrap: if True, add reduced flag to self
            new_flag: name of new flag, required if wrap is True
            remove_reduced: if True, remove all reduced flags from self
        
        Returns:
            out: reduced flag
        """
        if flags is None:
            ## copy needed to no break things if removing flags
            flags = self._fields.copy()

        to_reduce = [self._fields[f] for f in flags]
        if len(flags)==0:
            raise ValueError('Found zero flags to combine')
            
        out = self.get_zeros()
        
        ## need to add out to prevent flag ordering from causing errors
        ### (Ranges can't add to RangeMatrix, only other way around)
        to_reduce[0] = out+to_reduce[0]
        
        if method == 'union': 
            op = lambda x, y: x+y
        elif method == 'intersect':
            op = lambda x, y: x*y
        else:
            op = method
        out = reduce(op, to_reduce)
        
        # drop the fields if needed
        if remove_reduced: 
            for f in flags: 
                self.move(f, None)
                
        if wrap:
            if new_flag is None:
                raise ValueError("new_flag cannot be None if wrap is True")
            self.wrap(new_flag, out)
            
        return out        
    
    def has_cuts(self, flags=None):
        '''
        Return list of detector ids that have cuts
        
        Args: 
            flags: [optional] If not none it is the list of flags to combine to see
                    if cuts exist
        '''
        c = self.reduce(flags=flags)
        idx = np.where( [len(x.ranges())>0 for x in c])[0]
        return self[self._det_name].vals[idx]

    @classmethod
    def for_tod(cls, tod, det_name='dets', samp_name='samps'):
        """Assumes tod is an AxisManager with dets and samps axes defined
        """
        return cls(tod[det_name], tod[samp_name])

def _get_shape(data):
    try:
        return data.shape
    except:
        ### catches if a detector mask is just a list
        return np.shape(data)
    